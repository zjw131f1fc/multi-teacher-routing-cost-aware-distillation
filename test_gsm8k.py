"""测试 GSM8K 数据集加载。"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, '/data/users/zjw/projects/multi-teacher-routing-cost-aware-distillation')

from engine.configs import load_config
from engine.datas import load_dataset


def test_gsm8k_loading():
    """测试 GSM8K 数据集加载。"""

    print("=" * 80)
    print("测试 GSM8K 数据集加载")
    print("=" * 80)

    # 1. 首先加载配置（设置环境变量）
    print("\n1. 加载配置文件...")
    config_path = "configs/example_qa_gsm8k.yaml"

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 {config_path} 不存在")
        print("使用最小配置进行测试...")
        # 创建临时配置文件
        import yaml
        temp_config = {
            "global_settings": {
                "seed": 42,
                "device": "cuda",
                "dtype": "bfloat16",
                "dataset_cache_dir": "/tmp/dataset_cache",
                "hf_cache_dir": "/tmp/hf_cache",
            },
            "dataset_settings": {
                "type": "qa",
                "name": "qa-gsm8k",
                "qa_settings": {
                    "split": {
                        "train": 100,
                        "test": 50,
                    },
                    "hf_path": "openai/gsm8k",
                    "hf_config": "main",
                    "extract_final_answer": True,
                    "load_splits": ["train", "test"]
                }
            }
        }
        config_path = "/tmp/test_gsm8k_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(temp_config, f)
        print(f"✓ 创建临时配置文件: {config_path}")

    # 加载配置（这会设置环境变量）
    config = load_config(config_path)
    print("✓ 配置加载完成")

    # 2. 加载数据集
    print("\n2. 加载数据集...")
    dataset_bundle = load_dataset(config)
    print("✓ 数据集加载完成")

    # 3. 检查返回的结构
    print("\n3. 检查返回结构...")
    assert "splits" in dataset_bundle, "缺少 splits"
    assert "meta" in dataset_bundle, "缺少 meta"
    assert "judge" in dataset_bundle, "缺少 judge"
    print("✓ 返回结构正确")

    # 4. 检查 splits
    print("\n4. 检查 splits...")
    splits = dataset_bundle["splits"]
    assert "train" in splits, "缺少 train split"
    assert "test" in splits, "缺少 test split"
    print(f"✓ Train split 大小: {len(splits['train'])}")
    print(f"✓ Test split 大小: {len(splits['test'])}")

    # 5. 检查样本格式
    print("\n5. 检查样本格式...")
    train_sample = splits["train"][0]
    print(f"样本字段: {list(train_sample.keys())}")
    assert "question" in train_sample, "缺少 question"
    assert "answer" in train_sample, "缺少 answer"
    assert "final_answer" in train_sample, "缺少 final_answer"
    print("✓ 样本格式正确")

    # 6. 打印一个示例
    print("\n6. 示例样本:")
    print("-" * 80)
    print(f"Question: {train_sample['question'][:100]}...")
    print(f"Answer: {train_sample['answer'][:150]}...")
    print(f"Final Answer: {train_sample['final_answer']}")
    print("-" * 80)

    # 7. 测试 judge 函数
    print("\n7. 测试 judge 函数...")
    judge = dataset_bundle["judge"]

    # 测试单条判定
    result1 = judge("3", "3")
    print(f"judge('3', '3') = {result1}")
    assert result1["correct"] == 1, "判定错误"
    assert result1["accuracy"] == 1.0, "准确率错误"

    result2 = judge("3", "4")
    print(f"judge('3', '4') = {result2}")
    assert result2["correct"] == 0, "判定错误"
    assert result2["accuracy"] == 0.0, "准确率错误"

    # 测试批量判定
    result3 = judge(["3", "4", "5"], ["3", "5", "5"])
    print(f"judge(['3', '4', '5'], ['3', '5', '5']) = {result3}")
    assert result3["correct"] == 2, "批量判定错误"
    assert abs(result3["accuracy"] - 2/3) < 1e-5, "准确率错误"

    print("✓ Judge 函数工作正常")

    # 8. 检查元信息
    print("\n8. 元信息:")
    meta = dataset_bundle["meta"]
    print(f"Total samples: {meta['total']}")
    print(f"Split sizes: {meta['split_sizes']}")
    print(f"Split details: {meta['split_details']}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_gsm8k_loading()
