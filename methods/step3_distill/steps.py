"""训练和评估步骤函数（知识蒸馏）"""

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
    # 构建对话格式
    formatted_inputs = []
    for inp, tgt in zip(input_texts, target_texts):
        messages = [
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
    """评估步骤（知识蒸馏）

    参数:
        batch: 批次数据（原始数据，list of samples）
        device: 设备
        info: 评估信息

    返回:
        metrics: {
            "ce_loss": 交叉熵损失,
            "perplexity": 困惑度
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

        # 构建输入和目标
        input_texts.append(instruction)
        target_texts.append(teacher_response)

    if len(input_texts) == 0:
        return {
            "ce_loss": 0.0,
            "perplexity": 0.0
        }

    # Tokenize（使用 chat template）
    formatted_inputs = []
    for inp, tgt in zip(input_texts, target_texts):
        messages = [
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
