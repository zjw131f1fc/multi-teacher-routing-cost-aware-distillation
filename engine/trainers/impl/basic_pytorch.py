"""Basic PyTorch Trainer.

- 支持多优化器参数挂载
- 接收数据集并内部批处理
- 提供基本训练与评估循环

此训练器保持通用，不绑定具体模型；用户可通过
`register_train_step(name, fn)` 和 `register_eval_step(fn)`
注册具体的 step 计算逻辑，以实现灵活的训练流程。
"""

from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class _ParamGroupSpec:
    name: str
    params: List[torch.nn.Parameter]
    opt_type: str
    lr: float
    extra: Dict[str, Any]


class BasicPytorchTrainer:
    def __init__(self, config: Any, dataset_bundle: Dict[str, Any]):
        # 支持字典型 config 或具名属性访问
        self.config = config
        self.logger = getattr(config, "logger", None) or config.get("logger")

        # 训练相关设置（来自 configs/loader.py 的 DEFAULT_CONFIG.trainer_settings.dl_settings）
        if isinstance(config, dict):
            ts = config["trainer_settings"]["dl_settings"]
        else:
            ts = config.trainer_settings["dl_settings"] if isinstance(config.trainer_settings, dict) else config.trainer_settings.dl_settings
        self.epochs = int(ts["epochs"])
        self.batch_size = int(ts["batch_size"])
        # 配置项单位改为 batch
        self.print_loss_every_batches = int(ts["print_loss_every_batches"])  # 每隔多少个batch打印该batch的loss
        self.eval_every_batches = int(ts["eval_every_batches"])              # 每隔多少个batch评估一次（评估整个测试集）
        self.save_every_batches = int(ts["save_every_batches"])              # 每隔多少个batch保存一次checkpoint
        self.save_every_epochs = int(ts["save_every_epochs"])                # 每隔多少个epoch保存一次checkpoint
        self.eval_max_samples = int(ts["eval_max_samples"])                  # 每次评估抽样的最大样本数；<=0 表示全量
        
        # 梯度裁剪设置
        self.grad_clip_max_norm = ts.get("grad_clip_max_norm")  # None 表示不裁剪，float 表示裁剪阈值

        # Learning rate scheduler 设置
        self.use_warmup = ts.get("use_warmup", False)
        self.warmup_steps = int(ts.get("warmup_steps", 0))
        self.warmup_ratio = float(ts.get("warmup_ratio", 0.0))  # warmup占总步数的比例（如果warmup_steps为0）

        # 优化器设定
        self.optim_cfg = ts["optimizers"]
        self.param_groups: Dict[str, _ParamGroupSpec] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, Any] = {}  # Learning rate schedulers

        # 数据集
        self.dataset_bundle = dataset_bundle
        self.splits = dataset_bundle["splits"]
        self.meta = dataset_bundle["meta"]

        # 训练回调（单一）：返回字典，键为优化器组名，值为该组的 loss（或包含 'loss' 键的字典）
        # 约定: train_fn(batch, device, info) -> Dict[str, float | Dict[str, float]]
        self.train_fn: Optional[Callable[[List[Any], torch.device, Dict[str, int]], Dict[str, float]]] = None
        # - 评估: 单一评估函数，返回 Dict 指标
        #   约定: eval_fn(batch, device, info) -> Dict[str, float]
        self.eval_fn: Optional[Callable[[List[Any], torch.device, Dict[str, int]], Dict[str, float]]] = None

        # 模型注册表（用于在info中传递模型）
        self.models: Dict[str, torch.nn.Module] = {}

        # 设备
        if isinstance(config, dict):
            device_str = config["global_settings"]["device"]
        else:
            device_str = config.global_settings["device"] if isinstance(config.global_settings, dict) else getattr(config.global_settings, "device")
        self.device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

        # 训练状态跟踪（用于 train_batches 方法）
        self._current_epoch = 0
        self._current_epoch_batch_idx = 0
        self._global_batch_count = 0
        self._dataloader_iterator = None
        self._current_loader = None

        # 剪枝回调（用于Optuna实时剪枝）
        self.prune_callback = None  # 函数签名: (step, score) -> should_prune (bool)
        
        # 持久化存储（跨评估调用保持状态，如EMA score等）
        # 每个新的trial会重新创建trainer，所以这个字典会被重置
        self.persistent_state: Dict[str, Any] = {}

        # 训练历史跟踪（用于绘制曲线）
        self.training_history = {
            "epoch_losses": {},      # {loss_name: [epoch1_avg, epoch2_avg, ...]}
            "epoch_metrics": {},     # {metric_name: [epoch1_avg, epoch2_avg, ...]}
            "eval_history": [],      # [{epoch: int, metrics: {...}}, ...]
        }

        # 从cache获取Optuna状态池（如果有）
        if isinstance(dataset_bundle, dict):
            self._state_pool = dataset_bundle.get('_state_pool')
            self._cmd_queue = dataset_bundle.get('_cmd_queue')
        else:
            self._state_pool = None
            self._cmd_queue = None

        if self.logger:
            self.logger.info("BasicPytorchTrainer 初始化完成")
            self.logger.info(f"  - Device: {self.device}")
            self.logger.info(f"  - Epochs: {self.epochs}, Batch size: {self.batch_size}")

    # ---------------------- 参数与优化器管理 ----------------------
    def add_param_group(self, name: str, params: List[torch.nn.Parameter],
                        opt_type: Optional[str] = None, lr: Optional[float] = None, **extra):
        """注册一个参数组，稍后由 `setup_optimizers` 创建优化器。

        如果未指定 opt_type/lr，则从 config.trainer_settings.dl_settings.optimizers[name]
        中读取。支持 'adam' 与 'sgd'。
        """
        opt_cfg = self.optim_cfg.get(name)
        final_opt_type = (opt_type or opt_cfg["type"]).lower()
        final_lr = float(lr or opt_cfg["lr"])
        self.param_groups[name] = _ParamGroupSpec(name=name, params=list(params), opt_type=final_opt_type, lr=final_lr, extra=extra)
        if self.logger:
            self.logger.info(f"注册参数组: {name} | opt={final_opt_type} | lr={final_lr}")

    def setup_optimizers(self):
        """根据已注册的参数组创建优化器。"""
        for name, spec in self.param_groups.items():
            if spec.opt_type == "adam":
                weight_decay = float(spec.extra.get("weight_decay", 0.0))
                opt = torch.optim.Adam(spec.params, lr=spec.lr, weight_decay=weight_decay)
            elif spec.opt_type == "adamw":
                weight_decay = float(spec.extra.get("weight_decay", 0.01))
                opt = torch.optim.AdamW(spec.params, lr=spec.lr, weight_decay=weight_decay)
            elif spec.opt_type == "sgd":
                momentum_val = spec.extra["momentum"] if "momentum" in spec.extra else 0.0
                momentum = float(momentum_val)
                weight_decay = float(spec.extra.get("weight_decay", 0.0))
                opt = torch.optim.SGD(spec.params, lr=spec.lr, momentum=momentum, weight_decay=weight_decay)
            else:
                raise ValueError(f"不支持的优化器类型: {spec.opt_type}")
            self.optimizers[name] = opt
            if self.logger:
                self.logger.info(f"优化器就绪: {name} -> {spec.opt_type}")

    def setup_schedulers(self, total_steps: int):
        """设置 learning rate schedulers（在知道总步数后调用）

        参数:
            total_steps: 总训练步数
        """
        if not self.use_warmup:
            return

        # 计算 warmup 步数
        if self.warmup_steps > 0:
            warmup_steps = self.warmup_steps
        else:
            warmup_steps = int(total_steps * self.warmup_ratio)

        if warmup_steps == 0:
            if self.logger:
                self.logger.info("Warmup 步数为 0，不使用 warmup")
            return

        if self.logger:
            self.logger.info(f"设置 Warmup Scheduler: warmup_steps={warmup_steps}, total_steps={total_steps}")

        # 为每个优化器创建 scheduler
        for name, opt in self.optimizers.items():
            # 使用 Linear Warmup + Constant LR
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    # Warmup: 从 0 线性增长到 1
                    return float(current_step) / float(max(1, warmup_steps))
                # Warmup 后保持不变
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            self.schedulers[name] = scheduler
            if self.logger:
                self.logger.info(f"Scheduler 就绪: {name} -> Linear Warmup ({warmup_steps} steps)")

    # ---------------------- 模型注册 ----------------------
    def register_model(self, name: str, model):
        """注册模型（会在info中传递给train_step和eval_step）。
        
        参数:
            name: 模型名称
            model: PyTorch模型实例或其他对象
        """
        self.models[name] = model
        if self.logger:
            # 仅对torch.nn.Module打印参数量
            if isinstance(model, torch.nn.Module):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.logger.info(f"注册模型: {name} | 总参数量: {total_params:,} | 可训练参数量: {trainable_params:,}")
            else:
                self.logger.info(f"注册模型: {name}")
    
    # ---------------------- 训练/评估回调注册 ----------------------
    def register_train_step(self, fn: Callable[[List[Any], torch.device, Dict[str, Any]], Dict[str, float]]):
        """注册训练回调（单一）。
        返回字典：{ group_name: { 'loss_name1': value1, 'loss_name2': value2, ... } }
        trainer会自动将每个组的所有loss求和后反向传播
        回调参数包含 info: { 'config': config, 'epoch': int, 'batch': int, 'models': dict, ... }
        """
        self.train_fn = fn
        if self.logger:
            self.logger.info("注册训练回调完成")

    def register_eval_step(self, fn: Callable[[List[Any], torch.device, Dict[str, Any]], Dict[str, float]]):
        """注册评估 step 函数。
        fn 接口: fn(batch, device, info) -> Dict[str, float]
        info 包含 config、训练状态信息和models字典
        """
        self.eval_fn = fn
        if self.logger:
            self.logger.info("注册评估回调完成")

    # ---------------------- 内部 DataLoader ----------------------
    def _make_loader(self, dataset, shuffle: bool = True) -> DataLoader:
        def _collate_fn(batch):
            return batch  # 保持 list，不强制张量化
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=_collate_fn, num_workers=0, pin_memory=False)

    # ---------------------- 兼容路由层 ----------------------
    def run(self):
        """兼容路由层：根据配置决定使用train_full还是train_partial
        
        模式:
            - 普通模式：直接调用train_full()完整训练
            - Optuna模式：分批训练+定期评估+剪枝
        
        返回:
            - 普通模式：None
            - Optuna模式：Dict包含score、intermediate_results等
        """
        # 检测Optuna模式：优先检查manager_settings.mode
        if isinstance(self.config, dict):
            manager_settings = self.config.get("manager_settings", {})
            manager_mode = manager_settings.get("mode")
        else:
            manager_settings = getattr(self.config, "manager_settings", {})
            manager_mode = manager_settings.get("mode") if isinstance(manager_settings, dict) else None
        
        # 判断训练模式
        is_optuna_mode = (manager_mode == "optuna")
        
        if is_optuna_mode:
            if self.logger:
                self.logger.info("路由到Optuna超参数搜索模式")
            return self._run_optuna()
        else:
            if self.logger:
                self.logger.info("路由到完整训练模式")
            self.train_full()
            # 训练完成后做全测试
            eval_result = self.evaluate()
            return {"score": eval_result.get("score", 0), "eval_result": eval_result}
    
    def _run_optuna(self):
        """Optuna超参数搜索模式：分批训练+定期评估+实时剪枝"""
        if self.logger:
            self.logger.info("使用Optuna超参数搜索模式")
        
        # 获取配置参数
        if isinstance(self.config, dict):
            dl_settings = self.config["trainer_settings"]["dl_settings"]
        else:
            dl_settings = self.config.trainer_settings["dl_settings"]
        
        eval_interval = dl_settings.get("optuna_eval_interval_batches", 50)
        eval_max_samples = dl_settings.get("eval_max_samples", 50)
        
        if self.logger:
            self.logger.info(f"  - 评估间隔: 每 {eval_interval} 个batch")
            self.logger.info(f"  - 评估样本数: {eval_max_samples}")
        
        best_score = float('-inf')
        eval_step = 0
        intermediate_results = []
        
        while True:
            # 训练指定数量的batch
            train_result = self.train_partial(eval_interval)
            
            current_epoch = train_result.get("current_epoch", 0)
            total_batches = train_result.get("total_batches_trained", 0)
            training_completed = train_result.get("training_completed", False)
            
            # 执行评估
            max_samples = eval_max_samples if eval_max_samples > 0 else None
            eval_result = self.evaluate(max_samples=max_samples)
            
            # 检查评估结果是否包含score
            if "score" not in eval_result:
                available_keys = ", ".join(eval_result.keys()) if eval_result else "无"
                raise ValueError(
                    f"Optuna模式下，评估结果必须包含'score'字段！\n"
                    f"当前评估结果包含的字段: {available_keys}\n"
                    f"请在评估函数中添加'score'字段，例如：\n"
                    f"  return {{'accuracy': acc, 'score': acc}}"
                )
            
            score = eval_result["score"]
            
            # 更新最佳分数
            if score > best_score:
                best_score = score
            
            eval_step += 1
            
            # 记录中间结果
            intermediate_results.append({
                'step': total_batches,
                'score': score,
                'epoch': current_epoch
            })
            
            # 日志输出
            if self.logger:
                self.logger.info(f"[Optuna] Epoch {current_epoch}, Step {total_batches}, Eval #{eval_step}: score={score:.4f}, best={best_score:.4f}")
            
            # 实时剪枝：通过状态池上报中间结果
            if hasattr(self, '_state_pool') and self._state_pool and hasattr(self, '_cmd_queue'):
                # 更新状态池
                if self.logger:
                    self.logger.info(f"[Trainer] 上报INTERMEDIATE到状态池: step={total_batches}, score={score:.4f}")
                self._state_pool['intermediate_result'] = {
                    'step': total_batches,
                    'value': score
                }
                # 等待Manager的剪枝决策
                if self.logger:
                    self.logger.info(f"[Trainer] 等待PRUNE_DECISION...")
                while True:
                    decision_cmd = self._cmd_queue.get()
                    if decision_cmd.get('type') == 'PRUNE_DECISION':
                        should_prune = decision_cmd.get('data', False)
                        # 清空intermediate_result
                        self._state_pool['intermediate_result'] = None
                        if self.logger:
                            self.logger.info(f"[Trainer] 收到PRUNE_DECISION: {should_prune}")
                        if should_prune:
                            if self.logger:
                                self.logger.info(f"[Optuna] 在step {total_batches}被实时剪枝")
                            return {
                                'score': best_score,
                                'intermediate_results': intermediate_results,
                                'pruned': True
                            }
                        break
            elif self.prune_callback is not None:
                # 兼容旧的callback方式
                should_prune = self.prune_callback(total_batches, score)
                if should_prune:
                    if self.logger:
                        self.logger.info(f"[Optuna] 在step {total_batches}被实时剪枝")
                    return {
                        'score': best_score,
                        'intermediate_results': intermediate_results,
                        'pruned': True
                    }
            
            # 检查是否训练完成
            if training_completed:
                if self.logger:
                    self.logger.info(f"[Optuna] 训练完成，最终score={best_score:.4f}")
                # 清空状态池，避免manager重复处理
                if hasattr(self, '_state_pool') and self._state_pool:
                    self._state_pool['intermediate_result'] = None
                break
        
        # 返回结果时包含中间结果列表
        return {
            'score': best_score,
            'intermediate_results': intermediate_results,
            'pruned': False
        }
    
    # ---------------------- 训练循环 ----------------------
    def train_full(self):
        train_ds = self.splits["train"]
        if train_ds is None:
            raise ValueError("训练集不存在 (splits['train'] is None)")
        self.setup_optimizers()

        loader = self._make_loader(train_ds, shuffle=True)
        batch_count = 0
        total_planned_batches = self.epochs * len(loader)

        # 设置 schedulers（在知道总步数后）
        self.setup_schedulers(total_planned_batches)

        # 时间跟踪：用于预估剩余时间
        start_time = time.time()
        batch_times = []  # 记录最近若干batch的耗时

        if self.logger:
            self.logger.info(f"开始训练: epochs={self.epochs}, steps/epoch={len(loader)}")

        for epoch in range(self.epochs):
            if self.logger:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}")

            # 用于记录本epoch的所有loss
            epoch_losses = {}  # {group_name: [loss1, loss2, ...]}
            epoch_metrics = {}  # {metric_name: [val1, val2, ...]}

            for epoch_batch_idx, batch in enumerate(tqdm(loader, desc=f"Train E{epoch+1}")):
                batch_start_time = time.time()
                
                if self.train_fn is None:
                    raise RuntimeError("未注册训练回调")

                # 单次前向，返回各组的指标/损失
                info = {
                    "config": self.config,
                    "epoch": epoch + 1,
                    "batch": batch_count + 1,
                    "epoch_batch_index": epoch_batch_idx + 1,
                    "global_batch_index": batch_count + 1,
                    "total_planned_batches": total_planned_batches,
                    "models": self.models,
                    "persistent_state": self.persistent_state,
                }
                outputs = self.train_fn(batch, self.device, info)

                # 对每个优化器组进行反向与步进
                # 先清空所有优化器的梯度
                for opt in self.optimizers.values():
                    opt.zero_grad(set_to_none=True)
                
                # 然后分别 backward 各组的损失
                group_names = list(self.optimizers.keys())
                for idx, group_name in enumerate(group_names):
                    group_out = outputs[group_name]
                    # 如果是字典，将所有loss求和
                    if isinstance(group_out, dict):
                        loss = sum(group_out.values())
                    else:
                        loss = group_out
                    
                    if not torch.is_tensor(loss):
                        loss = torch.tensor(float(loss), dtype=torch.float32, device=self.device)
                    # 如果不是最后一个组，保留计算图
                    retain_graph = (idx < len(group_names) - 1)
                    loss.backward(retain_graph=retain_graph)
                
                # 梯度裁剪（在 backward 之后，optimizer.step 之前）
                if self.grad_clip_max_norm is not None:
                    for group_name, spec in self.param_groups.items():
                        torch.nn.utils.clip_grad_norm_(spec.params, self.grad_clip_max_norm)

                # === 收集 epoch loss 和 metrics ===
                metrics_info = outputs.get("metrics", {})
                for group_name in self.optimizers.keys():
                    group_out = outputs[group_name]
                    if isinstance(group_out, dict):
                        # 收集每个loss分量
                        for k, v in group_out.items():
                            loss_key = f"{group_name}.{k}"
                            loss_val = float(v.detach() if isinstance(v, torch.Tensor) else v)
                            if loss_key not in epoch_losses:
                                epoch_losses[loss_key] = []
                            epoch_losses[loss_key].append(loss_val)
                    else:
                        # 单个loss
                        loss_val = float(group_out.detach() if isinstance(group_out, torch.Tensor) else group_out)
                        if group_name not in epoch_losses:
                            epoch_losses[group_name] = []
                        epoch_losses[group_name].append(loss_val)

                # 收集 metrics
                for metric_name, metric_val in metrics_info.items():
                    if metric_name not in epoch_metrics:
                        epoch_metrics[metric_name] = []
                    epoch_metrics[metric_name].append(float(metric_val))

                # === 收集要打印的所有信息（梯度 + Loss + Metrics） ===
                if self.print_loss_every_batches and batch_count % self.print_loss_every_batches == 0:
                    # 1. 收集梯度信息
                    grad_stats = []
                    for group_name, spec in self.param_groups.items():
                        grad_norms = [p.grad.norm().item() for p in spec.params if p.grad is not None]
                        if grad_norms:
                            total_norm = (sum(g**2 for g in grad_norms))**0.5
                            grad_stats.append(f"grad_{group_name}={total_norm:.4f}")

                    # 2. 收集Loss信息
                    loss_parts = []
                    metrics_info = outputs.get("metrics", {})  # 提取 metrics

                    for group_name in self.optimizers.keys():
                        group_out = outputs[group_name]
                        if isinstance(group_out, dict):
                            loss_strs = [f"{k}={float(v.detach() if isinstance(v, torch.Tensor) else v):.4f}"
                                       for k, v in group_out.items()]
                            loss_parts.append(f"{group_name}[{', '.join(loss_strs)}]")
                        else:
                            loss_val = group_out.detach() if isinstance(group_out, torch.Tensor) else group_out
                            loss_parts.append(f"{group_name}={float(loss_val):.4f}")

                    # 3. 添加关键 metrics（保留率和判别器胜率）
                    metric_strs = []

                    # 每层的保留率 (L5_kept, L15_kept, L25_kept)
                    layer_metrics = {k: v for k, v in metrics_info.items() if k.startswith('L') and k.endswith('_kept')}
                    for layer_key in sorted(layer_metrics.keys(), key=lambda x: int(x[1:-5])):  # 按层号排序
                        metric_strs.append(f"{layer_key}={layer_metrics[layer_key]:.3f}")

                    # 最终保留率
                    if "final_kept_ratio" in metrics_info:
                        metric_strs.append(f"final={metrics_info['final_kept_ratio']:.3f}")

                    # 判别器胜率
                    if "disc_real_acc" in metrics_info and "disc_fake_acc" in metrics_info:
                        metric_strs.append(f"disc_R={metrics_info['disc_real_acc']:.3f}")
                        metric_strs.append(f"disc_F={metrics_info['disc_fake_acc']:.3f}")

                    # Top-1准确率（路由器）
                    if "top1_acc" in metrics_info:
                        metric_strs.append(f"top1={metrics_info['top1_acc']:.3f}")

                    # 4. 组装完整消息
                    all_parts = grad_stats + loss_parts + metric_strs

                    # 添加当前学习率（如果使用了scheduler）
                    if len(self.schedulers) > 0:
                        # 显示第一个优化器的学习率
                        first_opt_name = list(self.optimizers.keys())[0]
                        current_lr = self.optimizers[first_opt_name].param_groups[0]['lr']
                        all_parts.append(f"lr={current_lr:.2e}")

                    # 5. 计算ETA
                    remaining_batches = total_planned_batches - batch_count
                    if len(batch_times) > 0 and remaining_batches > 0:
                        avg_batch_time = sum(batch_times) / len(batch_times)
                        eta_seconds = avg_batch_time * remaining_batches
                        eta_str = self._format_time(eta_seconds)
                        all_parts.append(f"ETA: {eta_str}")

                    # 6. 统一打印
                    if self.logger:
                        self.logger.info(f"[Batch {batch_count}/{total_planned_batches}] " + " | ".join(all_parts))

                # 最后统一执行优化步骤
                for opt in self.optimizers.values():
                    opt.step()

                # 更新 learning rate scheduler
                for scheduler in self.schedulers.values():
                    scheduler.step()

                batch_count += 1

                # 记录本batch耗时
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                # 保留最近50个batch的时间用于平滑预估
                if len(batch_times) > 50:
                    batch_times.pop(0)

                # 每隔 N 个 batch 评估一次（评估整个测试集）
                if self.eval_every_batches and self.eval_fn and batch_count % self.eval_every_batches == 0:
                    # eval_max_samples <= 0 则评估全量测试集
                    max_samples = self.eval_max_samples if self.eval_max_samples > 0 else None
                    self.evaluate(max_samples=max_samples, trigger_info=info)

                # 每隔 N 个 batch 保存检查点
                if self.save_every_batches and batch_count % self.save_every_batches == 0:
                    self._save_checkpoint(epoch=epoch, batch=batch_count)

            # === Epoch 结束：打印平均 loss 和 metrics ===
            if self.logger and epoch_losses:
                import numpy as np

                # 计算平均 loss
                loss_summary = []
                for loss_name, loss_values in epoch_losses.items():
                    avg_loss = np.mean(loss_values)
                    loss_summary.append(f"{loss_name}={avg_loss:.4f}")

                    # 记录到历史
                    if loss_name not in self.training_history["epoch_losses"]:
                        self.training_history["epoch_losses"][loss_name] = []
                    self.training_history["epoch_losses"][loss_name].append(float(avg_loss))

                # 计算平均 metrics
                metric_summary = []
                for metric_name, metric_values in epoch_metrics.items():
                    avg_metric = np.mean(metric_values)
                    metric_summary.append(f"{metric_name}={avg_metric:.3f}")

                    # 记录到历史
                    if metric_name not in self.training_history["epoch_metrics"]:
                        self.training_history["epoch_metrics"][metric_name] = []
                    self.training_history["epoch_metrics"][metric_name].append(float(avg_metric))

                # 打印
                summary_parts = loss_summary + metric_summary
                self.logger.info(f"[Epoch {epoch+1} Summary] " + " | ".join(summary_parts))

            # 每隔 N 个 epoch 保存检查点
            if self.save_every_epochs and (epoch + 1) % self.save_every_epochs == 0:
                self._save_checkpoint(epoch=epoch, batch=None)

        # 训练结束：绘制并保存训练曲线
        if self.logger:
            self.logger.info("训练完成！正在绘制训练曲线...")
        self._plot_training_curves()

    # ---------------------- 评估循环 ----------------------
    def evaluate(self, max_samples: Optional[int] = None, trigger_info: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        if self.eval_fn is None:
            if self.logger:
                self.logger.info("未注册评估回调，跳过评估")
            return {}

        test_ds = self.splits["test"]
        if test_ds is None or len(test_ds) == 0:
            if self.logger:
                self.logger.info("测试集不存在或为空，跳过评估")
            return {}

        if max_samples is not None and max_samples > 0:
            # 随机抽样
            idx = torch.randperm(len(test_ds))[:min(len(test_ds), max_samples)].tolist()
            sub_ds = [test_ds[i] for i in idx]
        else:
            sub_ds = test_ds

        loader = self._make_loader(sub_ds, shuffle=False)
        agg: Dict[str, float] = {}
        # 专门累积桶分布统计
        bucket_dist_accum = {}  # {teacher_name: {"predicted_counts": [...], "ground_truth_counts": [...]}}
        n = 0
        for i, batch in enumerate(tqdm(loader, desc="Evaluate")):
            info = {
                "config": self.config,
                "epoch": (trigger_info["epoch"] if trigger_info and "epoch" in trigger_info else 0),
                "batch": (trigger_info.get("batch", 0) if trigger_info else 0),
                "epoch_batch_index": (trigger_info["epoch_batch_index"] if trigger_info and "epoch_batch_index" in trigger_info else 0),
                "global_batch_index": (trigger_info["global_batch_index"] if trigger_info and "global_batch_index" in trigger_info else 0),
                "total_planned_batches": (trigger_info["total_planned_batches"] if trigger_info and "total_planned_batches" in trigger_info else len(loader)),
                "eval_batch_index": i + 1,
                "eval_total_batches": len(loader),
                "models": self.models,
                "persistent_state": self.persistent_state,
            }
            metrics = self.eval_fn(batch, self.device, info)

            # 处理 teacher_bucket_dist（如果存在）
            if "teacher_bucket_dist" in metrics:
                teacher_dist = metrics.pop("teacher_bucket_dist")  # 从metrics中移除
                # 累积每个教师的桶counts
                for teacher_name, counts_dict in teacher_dist.items():
                    if teacher_name not in bucket_dist_accum:
                        bucket_dist_accum[teacher_name] = {
                            "predicted_counts": [0] * len(counts_dict["predicted_counts"]),
                            "ground_truth_counts": [0] * len(counts_dict["ground_truth_counts"])
                        }
                    # 累加counts
                    for j, count in enumerate(counts_dict["predicted_counts"]):
                        bucket_dist_accum[teacher_name]["predicted_counts"][j] += count
                    for j, count in enumerate(counts_dict["ground_truth_counts"]):
                        bucket_dist_accum[teacher_name]["ground_truth_counts"][j] += count

            # 处理其他标量指标
            for k, v in metrics.items():
                # 检查是否为标量（可以转换为float）
                try:
                    float_v = float(v)
                    # 标量指标：加和求平均
                    if k in agg:
                        agg[k] = agg[k] + float_v
                    else:
                        agg[k] = float_v
                except (TypeError, ValueError):
                    # 其他非标量指标：保留最后一个值
                    agg[k] = v
            n += 1

        # 标量指标求平均
        if n > 0:
            for k in list(agg.keys()):
                if isinstance(agg[k], (int, float)):
                    agg[k] /= n

        # 计算桶分布的百分比并添加到结果中
        if bucket_dist_accum:
            teacher_bucket_distributions = {}
            for teacher_name, counts_dict in bucket_dist_accum.items():
                pred_counts = counts_dict["predicted_counts"]
                true_counts = counts_dict["ground_truth_counts"]

                total_pred = sum(pred_counts)
                total_true = sum(true_counts)

                pred_percentages = [c / total_pred * 100 if total_pred > 0 else 0 for c in pred_counts]
                true_percentages = [c / total_true * 100 if total_true > 0 else 0 for c in true_counts]

                teacher_bucket_distributions[teacher_name] = {
                    "predicted": {
                        "counts": pred_counts,
                        "percentages": pred_percentages
                    },
                    "ground_truth": {
                        "counts": true_counts,
                        "percentages": true_percentages
                    }
                }
            agg["teacher_bucket_dist"] = teacher_bucket_distributions

        if self.logger:
            # 分离标量和非标量指标打印
            scalar_metrics = {k: v for k, v in agg.items() if k != "teacher_bucket_dist"}
            self.logger.info(f"评估完成 (标量指标): {scalar_metrics}")

            # 桶分布统计单独打印
            if "teacher_bucket_dist" in agg:
                self.logger.info("评估完成 (桶分布统计):")
                for teacher_name, dist in agg["teacher_bucket_dist"].items():
                    self.logger.info(f"  {teacher_name}:")
                    self.logger.info(f"    预测分布: counts={dist['predicted']['counts']}, percentages={[f'{p:.1f}%' for p in dist['predicted']['percentages']]}")
                    self.logger.info(f"    真实分布: counts={dist['ground_truth']['counts']}, percentages={[f'{p:.1f}%' for p in dist['ground_truth']['percentages']]}")

        # 记录评估历史（排除非标量值）
        if trigger_info and "epoch" in trigger_info:
            eval_record = {
                "epoch": trigger_info["epoch"],
                "metrics": {k: v for k, v in agg.items() if k != "teacher_bucket_dist" and isinstance(v, (int, float))}
            }
            self.training_history["eval_history"].append(eval_record)

        return agg

    # ---------------------- Checkpoint 保存 ----------------------
    def _save_checkpoint(self, epoch: int, batch: Optional[int]):
        # 基于自动生成的 experiment_tag 创建文件夹
        # 根目录来自 config.global_settings.save_dir
        if isinstance(self.config, dict):
            gs = self.config["global_settings"]
        else:
            gs = self.config.global_settings
        save_root = gs["save_dir"]
        tag = gs["experiment_tag"]

        import os
        out_dir = os.path.join(save_root, tag)
        os.makedirs(out_dir, exist_ok=True)

        # 文件名包含 epoch 与 batch（如有）
        if batch is not None:
            fname = f"epoch{epoch+1}_batch{batch}.pt"
        else:
            fname = f"epoch{epoch+1}.pt"
        path = os.path.join(out_dir, fname)

        # 保存优化器与参数组的状态（通用）
        state = {
            "epoch": epoch + 1,
            "batch": batch,
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "param_groups": {name: [p.detach().cpu() for p in spec.params] for name, spec in self.param_groups.items()},
        }
        torch.save(state, path)
        if self.logger:
            self.logger.info(f"Checkpoint 保存: {path}")
    
    # ---------------------- 训练指定batch数 ----------------------
    def train_partial(self, num_batches: int) -> Dict[str, Any]:
        """训练指定数量的batch，支持连续调用继续训练。
        
        该方法在训练过程中不会触发评估，只进行纯粹的训练。
        可以多次调用此方法，训练会从上次停止的地方继续。
        
        参数:
            num_batches: 要训练的batch数量
        
        返回:
            训练统计信息字典，包含:
            - 各优化器组的平均损失
            - batches_trained: 本次训练的batch数
            - total_batches_trained: 累计训练的总batch数
            - current_epoch: 当前所处的epoch
            - training_completed: 是否完成所有训练（达到epochs）
        """
        train_ds = self.splits["train"]
        if train_ds is None:
            raise ValueError("训练集不存在 (splits['train'] is None)")
        
        # 首次调用时初始化
        if self._current_loader is None:
            self.setup_optimizers()
            self._current_loader = self._make_loader(train_ds, shuffle=True)
            self._dataloader_iterator = iter(self._current_loader)
            if self.logger:
                self.logger.info(f"初始化训练: batch_size={self.batch_size}, steps/epoch={len(self._current_loader)}")
        
        if self.train_fn is None:
            raise RuntimeError("未注册训练回调")
        
        # 统计信息
        batch_losses = {name: [] for name in self.optimizers.keys()}
        batches_trained = 0
        
        if self.logger:
            self.logger.info(f"开始训练 {num_batches} 个batch (当前epoch={self._current_epoch+1}, 已训练batch={self._global_batch_count})")
        
        for _ in range(num_batches):
            try:
                if self._dataloader_iterator is None:
                    raise RuntimeError("DataLoader iterator 未初始化")
                batch = next(self._dataloader_iterator)
            except StopIteration:
                # 当前epoch结束，开始新的epoch
                self._current_epoch += 1
                self._current_epoch_batch_idx = 0
                
                # 每隔 N 个 epoch 保存检查点
                if self.save_every_epochs and self._current_epoch % self.save_every_epochs == 0:
                    self._save_checkpoint(epoch=self._current_epoch-1, batch=None)
                
                # 重新创建dataloader和iterator
                self._current_loader = self._make_loader(train_ds, shuffle=True)
                self._dataloader_iterator = iter(self._current_loader)
                
                if self.logger:
                    self.logger.info(f"Epoch {self._current_epoch} 完成，进入 Epoch {self._current_epoch+1}")
                
                batch = next(self._dataloader_iterator)
            
            # 执行训练步骤
            info = {
                "config": self.config,
                "epoch": self._current_epoch + 1,
                "batch": self._global_batch_count + 1,
                "epoch_batch_index": self._current_epoch_batch_idx + 1,
                "global_batch_index": self._global_batch_count + 1,
                "total_planned_batches": -1,  # 未知总数
                "models": self.models,
                "persistent_state": self.persistent_state,
            }
            outputs = self.train_fn(batch, self.device, info)
            
            # 梯度清零
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            
            # 反向传播
            group_names = list(self.optimizers.keys())
            for idx, group_name in enumerate(group_names):
                group_out = outputs[group_name]
                # 如果是字典，将所有loss求和
                if isinstance(group_out, dict):
                    loss = sum(group_out.values())
                else:
                    loss = group_out
                
                if not torch.is_tensor(loss):
                    loss = torch.tensor(float(loss), dtype=torch.float32, device=self.device)
                
                # 记录损失（先detach避免警告）
                batch_losses[group_name].append(loss.detach().item())
                
                # 保留计算图（除了最后一个组）
                retain_graph = (idx < len(group_names) - 1)
                loss.backward(retain_graph=retain_graph)
            
            # 优化步骤
            for opt in self.optimizers.values():
                opt.step()
            
            # 更新计数器
            self._current_epoch_batch_idx += 1
            self._global_batch_count += 1
            batches_trained += 1
            
            # 每隔 N 个 batch 保存检查点
            if self.save_every_batches and self._global_batch_count % self.save_every_batches == 0:
                self._save_checkpoint(epoch=self._current_epoch, batch=self._global_batch_count)
        
        # 计算平均损失
        avg_losses = {}
        for group_name, losses in batch_losses.items():
            if losses:
                avg_losses[f"{group_name}_loss"] = sum(losses) / len(losses)
        
        avg_losses["batches_trained"] = batches_trained
        avg_losses["total_batches_trained"] = self._global_batch_count
        avg_losses["current_epoch"] = self._current_epoch + 1
        # 检查是否完成所有训练（达到配置的epochs）
        avg_losses["training_completed"] = self._current_epoch >= self.epochs
        
        if self.logger:
            msg_parts = [f"{k}={v:.4f}" for k, v in avg_losses.items() if k.endswith("_loss")]
            self.logger.info(f"训练完成 {batches_trained} 个batch: " + ", ".join(msg_parts))
        
        return avg_losses
    
    # ---------------------- 辅助方法 ----------------------
    def _format_time(self, seconds: float) -> str:
        """格式化秒数为可读的时间字符串"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"

    def _plot_training_curves(self):
        """绘制并保存训练曲线到输出目录"""
        import os
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt

        # 获取输出目录
        if isinstance(self.config, dict):
            gs = self.config["global_settings"]
        else:
            gs = self.config.global_settings
        save_root = gs["save_dir"]
        tag = gs["experiment_tag"]
        out_dir = os.path.join(save_root, tag)
        os.makedirs(out_dir, exist_ok=True)

        epoch_losses = self.training_history["epoch_losses"]
        epoch_metrics = self.training_history["epoch_metrics"]
        eval_history = self.training_history["eval_history"]

        if not epoch_losses and not epoch_metrics and not eval_history:
            if self.logger:
                self.logger.info("没有训练历史数据，跳过绘图")
            return

        epochs = list(range(1, len(next(iter(epoch_losses.values()))) + 1)) if epoch_losses else []

        # === 绘制 Loss 曲线 ===
        if epoch_losses:
            plt.figure(figsize=(12, 6))
            for loss_name, loss_values in epoch_losses.items():
                plt.plot(epochs, loss_values, marker='o', label=loss_name, linewidth=2, markersize=4)

            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            loss_plot_path = os.path.join(out_dir, 'training_loss.png')
            plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            if self.logger:
                self.logger.info(f"Loss曲线已保存: {loss_plot_path}")

        # === 绘制 Metrics 曲线 ===
        if epoch_metrics:
            plt.figure(figsize=(12, 6))
            for metric_name, metric_values in epoch_metrics.items():
                plt.plot(epochs, metric_values, marker='s', label=metric_name, linewidth=2, markersize=4)

            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Metric Value', fontsize=12)
            plt.title('Training Metrics Curves', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            metrics_plot_path = os.path.join(out_dir, 'training_metrics.png')
            plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            if self.logger:
                self.logger.info(f"Metrics曲线已保存: {metrics_plot_path}")

        # === 绘制评估历史曲线 ===
        if eval_history:
            # 按指标分组
            eval_metrics = {}
            eval_epochs = []

            for record in eval_history:
                eval_epochs.append(record["epoch"])
                for metric_name, metric_value in record["metrics"].items():
                    if metric_name not in eval_metrics:
                        eval_metrics[metric_name] = []
                    eval_metrics[metric_name].append(metric_value)

            if eval_metrics:
                plt.figure(figsize=(12, 6))
                for metric_name, metric_values in eval_metrics.items():
                    plt.plot(eval_epochs, metric_values, marker='D', label=f"eval_{metric_name}",
                            linewidth=2, markersize=5, linestyle='--')

                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Evaluation Metric', fontsize=12)
                plt.title('Evaluation Metrics Curves', fontsize=14, fontweight='bold')
                plt.legend(loc='best', fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                eval_plot_path = os.path.join(out_dir, 'evaluation_metrics.png')
                plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
                plt.close()

                if self.logger:
                    self.logger.info(f"评估曲线已保存: {eval_plot_path}")
