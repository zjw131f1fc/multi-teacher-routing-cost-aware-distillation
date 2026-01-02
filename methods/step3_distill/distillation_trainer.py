"""自定义 HuggingFace Trainer，用于知识蒸馏

支持：
- 从蒸馏数据集中读取教师响应
- 使用路由策略选择最优教师
- **正确的 label masking**：只在教师回答部分计算损失
- 自定义 data collator 处理蒸馏数据格式
- 集成 GSM8K 评估（使用 judge 函数）
"""

import torch
from typing import Dict, Any, Optional, List
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput
from tqdm import tqdm


class DistillationTrainer(Trainer):
    """蒸馏训练的自定义 Trainer

    重写了 compute_loss 和 evaluation_loop 以适配蒸馏任务
    """

    def __init__(
        self,
        tokenizer,
        required_teachers: List[str],
        routing_strategy: str,
        best_teacher_name: Optional[str] = None,
        max_seq_length: int = 2048,
        router_model: Optional[Any] = None,
        knn_index: Optional[Any] = None,
        teacher_statistics: Optional[Dict] = None,
        dataset_bundle: Optional[Dict] = None,
        **kwargs
    ):
        """
        Args:
            tokenizer: 分词器
            required_teachers: 可用的教师列表
            routing_strategy: 路由策略 (random, best_teacher, model, knn_stats)
            best_teacher_name: 最强教师名称（strategy=best_teacher 时需要）
            max_seq_length: 最大序列长度
            router_model: 路由器模型（可选）
            knn_index: KNN 索引（可选）
            teacher_statistics: 教师统计信息（可选）
            dataset_bundle: 数据集 bundle（包含 judge 函数）
            **kwargs: 传递给 Trainer 的其他参数
        """
        super().__init__(**kwargs)

        self.custom_tokenizer = tokenizer
        self.required_teachers = required_teachers
        self.routing_strategy = routing_strategy
        self.best_teacher_name = best_teacher_name
        self.max_seq_length = max_seq_length
        self.router_model = router_model
        self.knn_index = knn_index
        self.teacher_statistics = teacher_statistics
        self.dataset_bundle = dataset_bundle

        # 导入路由函数
        from methods.step3_distill.router import route_to_teacher
        self.route_fn = route_to_teacher

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """自定义 loss 计算：从蒸馏数据中选择教师并训练

        **关键改进**：使用 label masking，只在教师回答部分计算损失

        Args:
            model: 学生模型
            inputs: 批次数据（list of samples）
            return_outputs: 是否返回模型输出
            num_items_in_batch: 批次中的样本数（HF Trainer 4.57.0+ 传递）

        Returns:
            loss: 标量损失
        """
        # inputs 是一个 batch（list of samples）
        batch = inputs

        # 准备训练数据
        input_texts = []
        target_texts = []

        # 获取当前 epoch（从 state 中获取）
        current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0

        for sample in batch:
            instruction = sample["instruction"]
            responses = sample.get("responses", {})

            # 使用路由函数选择教师
            selected_teacher = self.route_fn(
                instruction=instruction,
                required_teachers=self.required_teachers,
                epoch=current_epoch,
                strategy=self.routing_strategy,
                router_model=self.router_model,
                sample_data=sample,
                knn_index=self.knn_index,
                teacher_statistics=self.teacher_statistics,
                best_teacher_name=self.best_teacher_name
            )

            # 获取选中教师的响应
            if selected_teacher not in responses:
                continue

            teacher_data = responses[selected_teacher]
            messages = teacher_data.get("messages", [])

            # 提取教师的响应文本
            teacher_response = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    teacher_response = msg.get("content", "")
                    break

            if not teacher_response:
                continue

            # 去掉特殊标签（学生模型不是推理模型）
            teacher_response = teacher_response.replace("<cot>", "").replace("</cot>", "")
            teacher_response = teacher_response.replace("<think>", "").replace("</think>", "")
            teacher_response = teacher_response.replace("<thinking>", "").replace("</thinking>", "")

            # 构建输入和目标
            input_texts.append(instruction)
            target_texts.append(teacher_response)

        if len(input_texts) == 0:
            # 如果没有有效样本，返回零损失
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)

        # ==================== 构建带 label masking 的训练数据 ====================

        input_ids_list = []
        labels_list = []

        for inp, tgt in zip(input_texts, target_texts):
            # 1. 分别 tokenize prompt 和 response
            messages_prompt = [
                {"role": "system", "content": "You are a math expert. Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": inp}
            ]

            # Tokenize prompt 部分（不包含 assistant 回答）
            prompt_text = self.custom_tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True  # 添加 assistant 前缀
            )
            prompt_ids = self.custom_tokenizer.encode(prompt_text, add_special_tokens=False)

            # 2. Tokenize response 部分
            response_ids = self.custom_tokenizer.encode(tgt, add_special_tokens=False)

            # 3. 添加 EOS token
            eos_id = self.custom_tokenizer.eos_token_id

            # 4. 组合完整序列
            full_input_ids = prompt_ids + response_ids + [eos_id]

            # 5. 创建 labels：prompt 部分设为 -100（忽略），response 部分保留
            # -100 是 PyTorch CrossEntropyLoss 的 ignore_index
            labels = [-100] * len(prompt_ids) + response_ids + [eos_id]

            # 6. 截断到最大长度
            if len(full_input_ids) > self.max_seq_length:
                full_input_ids = full_input_ids[:self.max_seq_length]
                labels = labels[:self.max_seq_length]

            input_ids_list.append(full_input_ids)
            labels_list.append(labels)

        # 7. Padding 到相同长度
        max_len = max(len(ids) for ids in input_ids_list)
        pad_token_id = self.custom_tokenizer.pad_token_id

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            padding_length = max_len - len(input_ids)

            # Padding input_ids（左填充或右填充，这里使用右填充）
            padded_input_ids.append(input_ids + [pad_token_id] * padding_length)

            # Padding labels（pad 位置设为 -100）
            padded_labels.append(labels + [-100] * padding_length)

            # Attention mask（1 for real tokens, 0 for padding）
            attention_masks.append([1] * len(input_ids) + [0] * padding_length)

        # 8. 转换为 tensor
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=self.args.device)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.long, device=self.args.device)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=self.args.device)

        # 9. 前向传播
        outputs = model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            labels=labels_tensor  # 使用带 mask 的 labels
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """自定义评估循环：支持 QA 数据集评估（生成答案并使用 judge）"""

        # 检查是否有 judge 函数
        has_judge = self.dataset_bundle is not None and self.dataset_bundle.get("judge") is not None

        if not has_judge:
            # 没有 judge，使用默认评估（计算 loss）
            return super().evaluation_loop(
                dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
            )

        # 有 judge，执行 QA 评估（生成答案并判定）
        judge = self.dataset_bundle["judge"]

        # 准备问题和参考答案
        all_questions = []
        all_reference_answers = []

        for batch in dataloader:
            for sample in batch:
                question = sample["question"]
                final_answer = sample.get("final_answer", "")
                all_questions.append(question)
                all_reference_answers.append(final_answer)

        if len(all_questions) == 0:
            # 空数据集
            return EvalLoopOutput(
                predictions=None,
                label_ids=None,
                metrics={f"{metric_key_prefix}_accuracy": 0.0},
                num_samples=0,
            )

        # 生成答案
        self.model.eval()

        # 设置左填充（decoder-only 模型生成时需要）
        original_padding_side = self.custom_tokenizer.padding_side
        self.custom_tokenizer.padding_side = "left"

        # 分批生成
        gen_batch_size = 16  # 从 4 增加到 16，加快评估速度
        generated_answers = []
        truncated_count = 0

        num_batches = (len(all_questions) + gen_batch_size - 1) // gen_batch_size

        for i in tqdm(range(0, len(all_questions), gen_batch_size),
                      desc=description,
                      total=num_batches,
                      leave=False):
            batch_questions = all_questions[i:i + gen_batch_size]

            # 构建输入
            formatted_inputs = []
            for question in batch_questions:
                messages = [
                    {"role": "system", "content": "You are a math expert. Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": question}
                ]
                formatted_input = self.custom_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_inputs.append(formatted_input)

            # 批量 tokenize
            inputs = self.custom_tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            ).to(self.args.device)

            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,  # 改回 1024，避免截断（数学题推理过程较长）
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    use_cache=True,  # 显式启用 KV cache
                    pad_token_id=self.custom_tokenizer.pad_token_id,
                    eos_token_id=self.custom_tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )

            # 解码当前批次
            generated_sequences = outputs.sequences
            for idx, output in enumerate(generated_sequences):
                generated_text = self.custom_tokenizer.decode(output, skip_special_tokens=True)

                # 检查是否被截断
                input_length = inputs.input_ids[idx].shape[0]
                generated_length = output.shape[0] - input_length
                if generated_length >= 1024:  # 更新阈值为 1024
                    truncated_count += 1

                # 提取助手的回复
                if "assistant" in generated_text:
                    parts = generated_text.split("assistant")
                    if len(parts) > 1:
                        answer = parts[-1].strip()
                    else:
                        answer = generated_text
                else:
                    answer = generated_text

                generated_answers.append(answer)

        # 恢复原始 padding side
        self.custom_tokenizer.padding_side = original_padding_side

        # 使用 judge 函数判定
        result = judge(generated_answers, all_reference_answers)

        # 计算截断率
        truncation_rate = truncated_count / len(generated_answers) if len(generated_answers) > 0 else 0.0

        # 构建指标
        metrics = {
            f"{metric_key_prefix}_accuracy": result["accuracy"],
            f"{metric_key_prefix}_correct": result["correct"],
            f"{metric_key_prefix}_total": result["total"],
            f"{metric_key_prefix}_truncation_rate": truncation_rate,
            f"{metric_key_prefix}_truncated_count": truncated_count,
        }

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(all_questions),
        )
