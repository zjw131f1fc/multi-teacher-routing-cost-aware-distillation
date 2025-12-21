这是一个非常棒的工程化思路。将连续的回归值转化为**离散的“档位”（Bins）分类问题**，在深度学习中通常比直接回归更稳定（类似于 Distributional RL 中的 C51 算法），而且非常适合作为 DP（动态规划）控制器的输入。

针对你的需求（PerSyn 理念 + 归一化 + 10个档位），我为你设计了一个名为 **NTE (Normalized Teaching Efficacy)** 的指标体系，以及相应的路由模型训练与推断方案。

---

### 第一部分：核心 Score 设计 (NTE)

我们需要构造一个 $\text{Score} \in [0, 1]$，它既要符合 PerSyn 的发现（不能太难），又要符合蒸馏的训练需求（不能太简单/无梯度），且必须由 Quality（质量）主导。

建议公式如下：

$$ \text{NTE}(x, y) = \underbrace{\mathcal{V}(y)}_{\text{Quality}} \cdot \underbrace{\mathcal{M}_{\text{prox}}(y, S_\theta)}_{\text{Learnability Adjustment}} $$

#### 1. $\mathcal{V}(y) \in [0, 1]$：质量项（Quality）
直接使用 Reward Model 的归一化分数，或者基于规则（如代码通过率 0/1）。这是 PerSyn 中强调的“主导因素”。

#### 2. $\mathcal{M}_{\text{prox}} \in [0, 1]$：近端可学习项（Learnability）
为了解决“过难”和“过易”的问题，我们使用一个非对称的 **Beta 分布形态**函数，而非简单的 Loss。

设 $p = P_{student}(y|x)$ 为学生生成该回复的**长度归一化似然概率**（Geometric Mean Likelihood），范围在 $[0, 1]$。

$$ \mathcal{M}_{\text{prox}} = \frac{p^\alpha \cdot (1-p)^\beta}{Z} $$

*   **$p$ (Likelihood)**：代表 PerSyn 的 "Learnability"。$p$ 越小说明越难（Gap 越大）。
*   **$1-p$ (Uncertainty)**：代表信息增益潜力。$p \to 1$ 说明学生已经会了，不需要浪费强教师预算。
*   **$\alpha, \beta$ (超参数)**：控制形态。
    *   建议设 **$\alpha=2.0, \beta=0.5$**。
    *   **解读**：$\alpha > \beta$ 意味着我们更倾向于 $p$ 较大的区域（即符合 PerSyn 的观点：容易学的样本更好），但 $\beta$ 保证了当 $p=1$ 时分数归零（避免无效训练）。
*   **$Z$ (归一化常数)**：使得该项最大值为 1。对于给定 $\alpha, \beta$，这是一个常数。

---

### 第二部分：离散化与 Label 构造

既然你的路由模型要在“10个档位”里选择，我们需要将计算出的连续 $S_{raw}$ 映射到离散的类别标签。

#### 1. 动态分桶策略 (Quantile Binning)
**不要使用线性分桶**（例如 0-0.1 是一档，0.9-1.0 是一档）。因为在实际数据中，大部分样本的有效分数可能集中在 0.2-0.6 之间，线性分桶会导致大量样本挤在中间几个档位，无法区分优劣。

**建议做法**：
1.  在预处理阶段，计算出一个 Batch 或一部分数据的 NTE 分数分布。
2.  计算 **分位数 (Quantiles)**：$q_{10}, q_{20}, \dots, q_{90}$。
3.  定义 10 个档位 $k \in \{0, \dots, 9\}$，每个档位代表的**价值中心** $v_k$ 为该分桶的均值。

#### 2. 标签平滑 (Label Smoothing / Soft Labels)
为了让模型学到档位之间的关系（比如档位 8 比 档位 2 好，档位 8 和 档位 9 很接近），**不要使用 One-Hot Label**。

建议使用 **高斯平滑标签 (Gaussian Soft Labels)** 或者 **Two-Hot Encoding**。
假设某个样本计算出的真实分数为 0.83，它落在第 8 档（中心值 0.8）和第 9 档（中心值 0.9）之间。
*   **Label**: `[0, 0, 0, 0, 0, 0, 0, 0.7, 0.3]` (示例)
*   这样路由模型不仅能学到“选哪个档”，还能学到分布的不确定性。

---

### 第三部分：路由模型设计与推断 (The Router)

你的路由模型 $R(x)$ 输出的是 10 个 logits。

#### 1. 模型输出
$$ \mathbf{z} = R(x) \in \mathbb{R}^{10} $$
$$ \mathbf{probs} = \text{Softmax}(\mathbf{z}) = [w_0, w_1, \dots, w_9] $$

其中 $w_k$ 代表“该样本属于第 $k$ 档价值”的概率。

#### 2. 还原期望价值 (Expected Utility)
为了给后续的 **DP Controller (动态规划控制器)** 提供一个标量值来进行排序和背包规划，你需要计算分布的**期望值**，而不是取 argmax。

$$ \hat{U}_{pred} = \sum_{k=0}^{9} w_k \cdot v_k $$

*   $v_k$：是第 $k$ 个档位代表的实际分数中心值（例如：档位 0 代表 0.05，档位 9 代表 0.95）。

**为什么要用期望值？**
*   如果模型预测 `档位0: 0.4`, `档位9: 0.6`（极度不确定），Argmax 会选档位 9，这很冒险。
*   期望值会给出 `0.4*0.05 + 0.6*0.95 = 0.59`。这能更准确地反映风险调整后的收益，让 DP 算法做出更稳健的预算分配。

---

### 总结：完整工作流

1.  **构造 Ground Truth (离线/构造数据时)**：
    *   对每个样本 $x$ 和教师输出 $y$，计算 **NTE 分数**：
        $$ S = \mathcal{V}(y) \cdot \frac{p^\alpha(1-p)^\beta}{Z} $$
    *   将所有 $S$ 根据分位数映射到 10 个桶中，生成类别标签 $target\_class \in \{0..9\}$。

2.  **训练 Router**：
    *   输入：Prompt $x$。
    *   输出：10 维 Logits。
    *   Loss：KL 散度（如果用 Soft Label）或 Cross Entropy（如果用 One-Hot）。

3.  **使用 Router (在线蒸馏时)**：
    *   Router 输出 10 个概率分布 $w$。
    *   计算期望效用 $\hat{U} = \sum w_k v_k$。
    *   **DP Controller** 使用这个 $\hat{U}$ 和教师成本 $C$ 进行全局规划，决定是否购买该教师的服务。

这个设计完美契合了 **PerSyn 的"强弱有别"**、**预算控制**以及**深度学习分类优于回归**的工程经验。