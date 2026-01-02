"""测试混合数据集（蒸馏训练 + GSM8K 测试）加载"""

import sys
sys.path.insert(0, '/data/users/zjw/projects/multi-teacher-routing-cost-aware-distillation')

from engine.configs import load_config
from engine.datas import load_dataset


def test_mixed_dataset():
    """测试混合数据集加载"""

    print("=" * 80)
    print("测试混合数据集（蒸馏训练 + GSM8K 测试）加载")
    print("=" * 80)

    # 1. 加载配置
    print("\n1. 加载配置...")
    config_path = "configs/step3_distill_with_gsm8k.yaml"
    config = load_config(override_file=config_path)
    print("✓ 配置加载完成")

    # 2. 加载混合数据集
    print("\n2. 加载混合数据集...")
    dataset_bundle = load_dataset(config)
    print("✓ 数据集加载完成")

    # 3. 检查结构
    print("\n3. 检查返回结构...")
    assert "splits" in dataset_bundle, "缺少 splits"
    assert "meta" in dataset_bundle, "缺少 meta"
    assert "judge" in dataset_bundle, "缺少 judge"
    print("✓ 返回结构正确")

    # 4. 检查 splits
    print("\n4. 检查 splits...")
    splits = dataset_bundle["splits"]
    print(f"可用的 splits: {list(splits.keys())}")

    assert "train" in splits, "缺少 train split"
    assert "test" in splits, "缺少 test split"

    print(f"✓ Train split 大小: {len(splits['train'])}")
    print(f"✓ Test split 大小: {len(splits['test'])}")

    # 5. 检查训练集样本格式（蒸馏格式）
    print("\n5. 检查训练集样本格式...")
    train_sample = splits["train"][0]
    print(f"训练样本字段: {list(train_sample.keys())}")

    assert "instruction" in train_sample, "缺少 instruction"
    assert "responses" in train_sample, "缺少 responses"
    assert "metadata" in train_sample, "缺少 metadata"

    print(f"✓ 教师数量: {len(train_sample['responses'])}")
    print(f"✓ 教师列表: {list(train_sample['responses'].keys())}")
    print("✓ 训练样本格式正确（蒸馏格式）")

    # 6. 检查测试集样本格式（QA 格式转换为蒸馏格式）
    print("\n6. 检查测试集样本格式...")
    test_sample = splits["test"][0]
    print(f"测试样本字段: {list(test_sample.keys())}")

    assert "instruction" in test_sample, "缺少 instruction"
    assert "metadata" in test_sample, "缺少 metadata"

    metadata = test_sample["metadata"]
    assert "final_answer" in metadata, "缺少 final_answer"
    assert "source" in metadata, "缺少 source"
    assert metadata["source"] == "gsm8k", "source 应为 gsm8k"

    print(f"✓ Question: {test_sample['instruction'][:80]}...")
    print(f"✓ Final Answer: {metadata['final_answer']}")
    print("✓ 测试样本格式正确（从 QA 转换）")

    # 7. 测试 judge 函数
    print("\n7. 测试 judge 函数...")
    judge = dataset_bundle["judge"]

    if judge is not None:
        # 测试单条判定
        result1 = judge("3", "3")
        print(f"judge('3', '3') = {result1}")
        assert result1["correct"] == 1, "判定错误"
        assert result1["accuracy"] == 1.0, "准确率错误"

        # 测试批量判定
        result2 = judge(["3", "4", "5"], ["3", "5", "5"])
        print(f"judge(['3', '4', '5'], ['3', '5', '5']) = {result2}")
        assert result2["correct"] == 2, "批量判定错误"

        print("✓ Judge 函数工作正常")
    else:
        print("⚠ Judge 函数为 None")

    # 8. 检查元信息
    print("\n8. 元信息:")
    meta = dataset_bundle["meta"]
    print(f"Total samples: {meta['total']}")
    print(f"Split sizes: {meta['split_sizes']}")
    print(f"Split details: {meta['split_details']}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)

    print("\n数据集摘要:")
    print(f"- 训练集（蒸馏数据）: {len(splits['train'])} 样本")
    print(f"- 测试集（GSM8K）: {len(splits['test'])} 样本")
    print(f"- Judge 函数: {'可用' if judge is not None else '不可用'}")


if __name__ == "__main__":
    test_mixed_dataset()
