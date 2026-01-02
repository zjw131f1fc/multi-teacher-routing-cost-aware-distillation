"""测试 GSM8K 数据集加载（离线版本，使用模拟数据）。"""

import sys
from types import SimpleNamespace

# 添加项目根目录到路径
sys.path.insert(0, '/data/users/zjw/projects/multi-teacher-routing-cost-aware-distillation')


def test_gsm8k_structure():
    """测试 GSM8K 数据集结构（不需要网络）。"""
    from engine.datas.base.qa import BasePreparer, QADataset

    print("=" * 80)
    print("测试 GSM8K 数据集结构")
    print("=" * 80)

    # 1. 测试基类功能
    print("\n1. 测试基类功能...")

    config = SimpleNamespace(
        dataset_settings=SimpleNamespace(
            type="qa",
            name="qa-gsm8k",
            qa_settings=SimpleNamespace(
                split={
                    "train": 0.8,
                    "test": 0.2,
                },
                extract_final_answer=True
            )
        ),
        global_settings=SimpleNamespace(
            seed=42
        ),
        logger=None
    )

    preparer = BasePreparer(config)
    print(f"✓ Split config: {preparer.split_cfg}")
    print(f"✓ Extract final answer: {preparer.extract_final_answer}")

    # 2. 测试答案提取
    print("\n2. 测试答案提取...")

    # GSM8K 格式
    answer_text = "The calculation shows 2+1=3\n#### 3"
    extracted = preparer.extract_answer(answer_text, format_type="gsm8k")
    print(f"Input: '{answer_text}'")
    print(f"Extracted: '{extracted}'")
    assert extracted == "3", f"提取错误: {extracted}"
    print("✓ GSM8K 格式答案提取正确")

    # MATH 格式
    answer_text2 = "Therefore the answer is \\boxed{42}"
    extracted2 = preparer.extract_answer(answer_text2, format_type="math")
    print(f"Input: '{answer_text2}'")
    print(f"Extracted: '{extracted2}'")
    assert extracted2 == "42", f"提取错误: {extracted2}"
    print("✓ MATH 格式答案提取正确")

    # 3. 测试 split 功能
    print("\n3. 测试 split 功能...")

    mock_samples = [
        {"question": f"Q{i}", "answer": f"A{i}", "final_answer": str(i)}
        for i in range(100)
    ]

    split_datasets, placeholder = preparer.split_samples(mock_samples)
    print(f"✓ Total samples: {len(mock_samples)}")
    print(f"✓ Train split: {len(split_datasets['train'])}")
    print(f"✓ Test split: {len(split_datasets['test'])}")
    print(f"✓ Placeholder splits: {placeholder}")

    assert isinstance(split_datasets['train'], QADataset)
    assert isinstance(split_datasets['test'], QADataset)
    assert len(split_datasets['train']) == 80  # 0.8 * 100
    assert len(split_datasets['test']) == 20   # 0.2 * 100

    # 4. 测试元信息构建
    print("\n4. 测试元信息构建...")

    meta = preparer.build_meta(mock_samples, split_datasets, placeholder)
    print(f"✓ Meta keys: {list(meta.keys())}")
    print(f"✓ Total: {meta['total']}")
    print(f"✓ Split sizes: {meta['split_sizes']}")
    print(f"✓ Split details: {meta['split_details']}")

    assert meta['total'] == 100
    assert meta['split_sizes']['train'] == 80
    assert meta['split_sizes']['test'] == 20

    # 5. 测试 judge 函数（模拟实现）
    print("\n5. 测试 judge 函数...")

    # 导入 GSM8K 的 judge 构建逻辑（简化版）
    def normalize_number(s):
        import re
        if s is None:
            return None
        s = str(s).replace(",", "").replace("$", "").strip()
        match = re.search(r'-?\d+\.?\d*', s)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    def simple_judge(pred, ref, sample=None):
        if isinstance(pred, list):
            total = len(pred)
            correct = 0
            for p, r in zip(pred, ref):
                p_num = normalize_number(str(p))
                r_num = normalize_number(str(r))
                if p_num is not None and r_num is not None and abs(p_num - r_num) < 1e-5:
                    correct += 1
            return {"correct": correct, "total": total, "accuracy": correct / total if total > 0 else 0.0}

        p_num = normalize_number(str(pred))
        r_num = normalize_number(str(ref))
        is_correct = (p_num is not None and r_num is not None and abs(p_num - r_num) < 1e-5)
        return {"correct": 1 if is_correct else 0, "total": 1, "accuracy": 1.0 if is_correct else 0.0}

    # 测试单条
    result1 = simple_judge("3", "3")
    print(f"judge('3', '3') = {result1}")
    assert result1["correct"] == 1
    assert result1["accuracy"] == 1.0

    result2 = simple_judge("3.0", "3")
    print(f"judge('3.0', '3') = {result2}")
    assert result2["correct"] == 1

    result3 = simple_judge("$3", "3")
    print(f"judge('$3', '3') = {result3}")
    assert result3["correct"] == 1

    # 测试批量
    result4 = simple_judge(["1", "2", "3"], ["1", "5", "3"])
    print(f"judge(['1', '2', '3'], ['1', '5', '3']) = {result4}")
    assert result4["correct"] == 2
    assert abs(result4["accuracy"] - 2/3) < 1e-5

    print("✓ Judge 函数工作正常")

    # 6. 验证注册表
    print("\n6. 验证注册表...")
    from engine.datas.loader import DATASET_REGISTRY

    assert "qa-gsm8k" in DATASET_REGISTRY, "GSM8K 未注册"
    print(f"✓ GSM8K 已注册: {DATASET_REGISTRY['qa-gsm8k']}")

    # 7. 验证所有 QA 数据集
    print("\n7. 已注册的 QA 数据集:")
    qa_datasets = [k for k in DATASET_REGISTRY.keys() if k.startswith("qa-")]
    for ds in qa_datasets:
        print(f"  - {ds}: {DATASET_REGISTRY[ds].__name__}")

    print("\n" + "=" * 80)
    print("✓ 所有离线测试通过！")
    print("=" * 80)
    print("\n注意: 在线测试需要网络连接到 HuggingFace。")
    print("你可以在有网络时使用以下代码测试:")
    print("""
    from engine.datas import load_dataset
    config = {...}  # 配置如上
    dataset_bundle = load_dataset(config)
    """)


if __name__ == "__main__":
    test_gsm8k_structure()
