"""测试 step3 的新多数据集配置"""

import sys
sys.path.insert(0, '/data/users/zjw/projects/multi-teacher-routing-cost-aware-distillation')

from engine.configs import load_config
from engine.datas import load_dataset


def test_step3_config():
    """测试 step3 新的多数据集配置"""
    print("=" * 80)
    print("测试 Step3 新的多数据集配置")
    print("=" * 80)

    # 加载配置
    config = load_config(override_file="configs/step3_distill_with_gsm8k.yaml")
    print(f"\n✓ 配置加载成功")

    # 检查 dataset_settings 是列表
    dataset_settings = config["dataset_settings"]
    print(f"\n✓ dataset_settings 类型: {type(dataset_settings)}")
    assert isinstance(dataset_settings, list), "dataset_settings 应该是列表"
    print(f"✓ dataset_settings 包含 {len(dataset_settings)} 个数据集配置")

    # 显示每个数据集的配置
    for idx, ds_config in enumerate(dataset_settings):
        print(f"\n数据集 {idx + 1}:")
        print(f"  - type: {ds_config['type']}")
        print(f"  - name: {ds_config['name']}")

    # 加载数据集
    print("\n" + "=" * 80)
    print("加载数据集...")
    print("=" * 80)

    dataset_bundles = load_dataset(config)

    # 验证返回类型
    print(f"\n✓ 返回类型: {type(dataset_bundles)}")
    assert isinstance(dataset_bundles, list), "多数据集应返回列表"
    print(f"✓ 返回 {len(dataset_bundles)} 个 bundle")

    # 分离两个数据集
    distill_bundle = dataset_bundles[0]
    qa_bundle = dataset_bundles[1]

    # 检查蒸馏数据集
    print("\n--- 蒸馏数据集 (OpenR1-Math) ---")
    print(f"✓ Dataset: {distill_bundle['meta'].get('dataset_name', 'N/A')}")
    print(f"✓ Splits: {list(distill_bundle['splits'].keys())}")
    for split_name, split_data in distill_bundle["splits"].items():
        print(f"  - {split_name}: {len(split_data)} 样本")

    # 检查 QA 数据集
    print("\n--- QA 数据集 (GSM8K) ---")
    print(f"✓ Dataset: {qa_bundle['meta'].get('dataset_name', 'N/A')}")
    print(f"✓ Splits: {list(qa_bundle['splits'].keys())}")
    for split_name, split_data in qa_bundle["splits"].items():
        print(f"  - {split_name}: {len(split_data)} 样本")

    # 检查 judge 函数
    print("\n--- Judge 函数 ---")
    print(f"✓ 蒸馏数据集 judge: {distill_bundle.get('judge')}")
    print(f"✓ QA 数据集 judge: {qa_bundle.get('judge')}")

    # 检查样本格式
    print("\n--- 样本格式检查 ---")

    # 蒸馏数据集样本
    if len(distill_bundle["splits"]["train"]) > 0:
        distill_sample = distill_bundle["splits"]["train"][0]
        print(f"✓ 蒸馏样本字段: {list(distill_sample.keys())}")
        print(f"  - instruction: {distill_sample.get('instruction', 'N/A')[:50]}...")
        print(f"  - responses: {list(distill_sample.get('responses', {}).keys())}")

    # QA 数据集样本
    if len(qa_bundle["splits"]["test"]) > 0:
        qa_sample = qa_bundle["splits"]["test"][0]
        print(f"✓ QA 样本字段: {list(qa_sample.keys())}")
        if "metadata" in qa_sample:
            print(f"  - metadata: {list(qa_sample['metadata'].keys())}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        test_step3_config()
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
