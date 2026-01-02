"""训练和评估步骤函数（知识蒸馏，支持 QA 测试集评估）"""

import torch
import torch.nn.functional as F
from typing import Dict, Any


def train_step(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """训练步骤（知识蒸馏）

    参数:
        batch: 批次数据，每个样本格式为:
            {
                "instruction": str,
                "responses": {
                    "teacher_name": {
                        "messages": [...],
                        "nte_scores": {...}
                    }
                }
            }
        device: 设备
        info: 训练信息，包含:
            - models: 注册的模型字典（student, tokenizer, required_teachers, router_model）
            - config: 配置对象
            - epoch: 当前训练轮次

    返回:
        losses: {
            "student": {
                "loss": 总损失,
                "ce_loss": 交叉熵损失
            },
            "metrics": {
                "perplexity": 困惑度
            }
        }
    """
    # 获取模型和配置
    student_model = info["models"]["student"]
    tokenizer = info["models"]["tokenizer"]
    required_teachers = info["models"]["required_teachers"]
    router_model = info["models"].get("router_model", None)
    knn_index = info["models"].get("knn_index", None)
    teacher_statistics = info["models"].get("teacher_statistics", None)
    config = info["config"]
    epoch = info.get("epoch", 0)  # 获取当前epoch

    # 从配置中获取参数
    max_seq_length = config["method_settings"].get("max_seq_length", 2048)
    routing_strategy = config["method_settings"].get("routing_strategy", "random")
    best_teacher_name = config["method_settings"].get("best_teacher_name", None)

    # 导入路由函数
    from .router import route_to_teacher

    # 准备训练数据
    input_texts = []
    target_texts = []

    for sample in batch:
        instruction = sample["instruction"]
        responses = sample.get("responses", {})

        # 使用路由函数选择教师
        selected_teacher = route_to_teacher(
            instruction=instruction,
            required_teachers=required_teachers,
            epoch=epoch,
            strategy=routing_strategy,
            router_model=router_model,
            sample_data=sample,
            knn_index=knn_index,
            teacher_statistics=teacher_statistics,
            best_teacher_name=best_teacher_name
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

        # 去掉 <cot> 和 </cot> 标签（学生模型不是推理模型）
        teacher_response = teacher_response.replace("<cot>", "").replace("</cot>", "")

        # 构建输入和目标
        # 输入: instruction
        # 目标: teacher_response
        input_texts.append(instruction)
        target_texts.append(teacher_response)

    if len(input_texts) == 0:
        # 如果没有有效样本，返回零损失
        return {
            "student": {
                "loss": torch.tensor(0.0, device=device),
                "ce_loss": 0.0
            },
            "metrics": {
                "perplexity": 0.0
            }
        }

    # Tokenize（使用 chat template）
    # 构建对话格式，添加 system prompt 引导模型输出格式
    formatted_inputs = []
    for inp, tgt in zip(input_texts, target_texts):
        messages = [
            {"role": "system", "content": "You are a math expert. Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": tgt}
        ]
        formatted_inputs.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        ))

    # Tokenize
    encodings = tokenizer(
        formatted_inputs,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # 前向传播
    outputs = student_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids  # 自回归语言模型：labels = input_ids
    )

    # 计算损失（HuggingFace模型自动计算）
    loss = outputs.loss

    # 计算困惑度
    with torch.no_grad():
        perplexity = torch.exp(loss).item()

    return {
        "student": {
            "loss": loss,
            "ce_loss": loss.item()
        },
        "metrics": {
            "perplexity": perplexity
        }
    }


def eval_step(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """评估步骤（支持 QA 测试集评估）

    参数:
        batch: 批次数据（原始数据，list of samples）
        device: 设备
        info: 评估信息

    返回:
        metrics: {
            "ce_loss": 交叉熵损失,
            "perplexity": 困惑度,
            "accuracy": 准确率（如果有 QA judge）,
            "correct": 正确数量,
            "total": 总数量
        }
    """
    # 获取模型和配置
    student_model = info["models"]["student"]
    tokenizer = info["models"]["tokenizer"]
    config = info["config"]

    # 从配置中获取参数
    max_seq_length = config["method_settings"].get("max_seq_length", 2048)

    # 检查是否是 QA 数据（有 final_answer 字段）
    is_qa_data = False
    if len(batch) > 0:
        first_sample = batch[0]
        # QA 数据直接有 final_answer 字段（不在 metadata 中）
        is_qa_data = "final_answer" in first_sample

    if is_qa_data:
        # QA 评估：生成答案并使用 judge 判定
        return _eval_qa(batch, device, info, student_model, tokenizer, max_seq_length)
    else:
        # 蒸馏评估：计算损失和困惑度
        return _eval_distill(batch, device, info, student_model, tokenizer, max_seq_length)


def _eval_distill(batch, device, info, student_model, tokenizer, max_seq_length):
    """蒸馏数据评估：计算损失和困惑度"""
    required_teachers = info["models"]["required_teachers"]
    router_model = info["models"].get("router_model", None)
    knn_index = info["models"].get("knn_index", None)
    teacher_statistics = info["models"].get("teacher_statistics", None)
    config = info["config"]
    epoch = info.get("epoch", 0)

    routing_strategy = config["method_settings"].get("routing_strategy", "random")
    best_teacher_name = config["method_settings"].get("best_teacher_name", None)

    from .router import route_to_teacher

    # 准备评估数据
    input_texts = []
    target_texts = []

    for sample in batch:
        instruction = sample["instruction"]
        responses = sample.get("responses", {})

        # 使用路由函数选择教师
        selected_teacher = route_to_teacher(
            instruction=instruction,
            required_teachers=required_teachers,
            epoch=epoch,
            strategy=routing_strategy,
            router_model=router_model,
            sample_data=sample,
            knn_index=knn_index,
            teacher_statistics=teacher_statistics,
            best_teacher_name=best_teacher_name
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

        # 去掉常见的特殊标签（学生模型不是推理模型）
        teacher_response = teacher_response.replace("<think>", "").replace("</think>", "")
        teacher_response = teacher_response.replace("<cot>", "").replace("</cot>", "")
        teacher_response = teacher_response.replace("<thinking>", "").replace("</thinking>", "")

        # 构建输入和目标
        input_texts.append(instruction)
        target_texts.append(teacher_response)

    if len(input_texts) == 0:
        return {
            "ce_loss": 0.0,
            "perplexity": 0.0
        }

    # Tokenize（使用 chat template）
    # 添加 system prompt 引导模型输出格式
    formatted_inputs = []
    for inp, tgt in zip(input_texts, target_texts):
        messages = [
            {"role": "system", "content": "You are a math expert. Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": tgt}
        ]
        formatted_inputs.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        ))

    # Tokenize
    encodings = tokenizer(
        formatted_inputs,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # 前向传播
    with torch.no_grad():
        outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

    # 计算损失
    loss = outputs.loss.item()

    # 计算困惑度
    perplexity = torch.exp(torch.tensor(loss)).item()

    return {
        "ce_loss": loss,
        "perplexity": perplexity
    }


def _eval_qa(batch, device, info, student_model, tokenizer, max_seq_length):
    """QA 数据评估：生成答案并使用 judge 判定"""
    logger = getattr(info["config"], "logger", None)

    # 导入 tqdm
    from tqdm import tqdm

    # 获取 judge 函数
    dataset_bundle = info["models"].get("dataset_bundle")
    if dataset_bundle is None or dataset_bundle.get("judge") is None:
        if logger:
            logger.warning("[Eval] No judge function found, falling back to distill eval")
        return _eval_distill(batch, device, info, student_model, tokenizer, max_seq_length)

    judge = dataset_bundle["judge"]

    # 准备问题和参考答案
    questions = []
    reference_answers = []

    for sample in batch:
        # QA 数据使用 question 和 final_answer 字段
        question = sample["question"]
        final_answer = sample.get("final_answer", "")

        questions.append(question)
        reference_answers.append(final_answer)

    if len(questions) == 0:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0
        }

    # 生成答案（分批处理以显示进度）
    student_model.eval()

    # 构建所有输入
    formatted_inputs = []
    for question in questions:
        messages = [
            {"role": "system", "content": "You are a math expert. Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question}
        ]
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_inputs.append(formatted_input)

    # 设置左填充（decoder-only 模型生成时需要）
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # 分批生成（每次生成固定数量以显示进度）
    gen_batch_size = 4  # 每次生成 4 个样本
    generated_answers = []

    num_batches = (len(formatted_inputs) + gen_batch_size - 1) // gen_batch_size

    for i in tqdm(range(0, len(formatted_inputs), gen_batch_size),
                  desc="Generating answers",
                  total=num_batches,
                  leave=False):
        batch_inputs = formatted_inputs[i:i + gen_batch_size]

        # 批量 tokenize
        inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length
        ).to(device)

        # 批量生成
        with torch.no_grad():
            outputs = student_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # 贪婪解码
                temperature=None,  # 消除警告
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 解码当前批次
        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)

            # 提取助手的回复（去掉用户输入部分）
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
    tokenizer.padding_side = original_padding_side

    # 使用 judge 函数判定
    result = judge(generated_answers, reference_answers)

    return {
        "accuracy": result["accuracy"],
        "correct": result["correct"],
        "total": result["total"]
    }
