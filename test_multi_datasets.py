"""测试多数据集加载功能"""

import sys
sys.path.insert(0, '/data/users/zjw/projects/multi-teacher-routing-cost-aware-distillation')

from engine.configs import load_config
from engine.datas import load_dataset


def test_single_dataset():
    """测试单数据集加载（向后兼容）"""
    print("=" * 80)
    print("测试 1: 单数据集加载（向后兼容）")
    print("=" * 80)

    # 使用现有的配置文件
    config = load_config(override_file="configs/step3_distill_with_gsm8k.yaml")
    print(f"\n加载配置: {config['dataset_settings']['name']}")

    dataset_bundle = load_dataset(config)

    # 验证返回格式
    print(f"✓ 返回类型: {type(dataset_bundle)}")
    assert isinstance(dataset_bundle, dict), "单数据集应返回 dict"
    assert "splits" in dataset_bundle, "缺少 splits"
    assert "meta" in dataset_bundle, "缺少 meta"
    assert "judge" in dataset_bundle, "缺少 judge"
    print(f"✓ 返回结构正确: {list(dataset_bundle.keys())}")

    # 检查 splits
    splits = dataset_bundle["splits"]
    print(f"✓ Splits: {list(splits.keys())}")
    print(f"  - train: {len(splits.get('train', []))} 样本")
    print(f"  - test: {len(splits.get('test', []))} 样本")

    print("\n✓ 单数据集加载测试通过！\n")
    return True


def test_multi_datasets_dict():
    """测试多数据集加载（使用 dict 配置）"""
    print("=" * 80)
    print("测试 2: 多数据集加载（dict 配置）")
    print("=" * 80)

    # 手动创建多数据集配置
    config = {
        "global_settings": {
            "seed": 42,
            "device": "cuda",
            "dtype": "bfloat16",
            "dataset_cache_dir": "/data/users/zjw/dataset_cache",
            "hf_cache_dir": "/data/users/zjw/huggingface_cache",
        },
        "dataset_settings": [
            # 数据集 1: 蒸馏数据集
            {
                "type": "distill",
                "name": "distill-openr1-math",
                "distill_settings": {
                    "split": {
                        "train": 100,  # 少量样本用于测试
                        "test": -1
                    }
                }
            },
            # 数据集 2: QA 数据集
            {
                "type": "qa",
                "name": "qa-gsm8k",
                "qa_settings": {
                    "split": {
                        "train": -1,
                        "test": 50  # 少量样本用于测试
                    }
                }
            }
        ],
        "logger": None
    }

    print("\n配置包含 2 个数据集:")
    for idx, ds in enumerate(config["dataset_settings"]):
        print(f"  {idx + 1}. {ds['name']}")

    bundles = load_dataset(config)

    # 验证返回格式
    print(f"\n✓ 返回类型: {type(bundles)}")
    assert isinstance(bundles, list), "多数据集应返回 list"
    assert len(bundles) == 2, f"应返回 2 个 bundle，实际: {len(bundles)}"
    print(f"✓ 返回 {len(bundles)} 个 bundle")

    # 检查每个 bundle
    for idx, bundle in enumerate(bundles):
        print(f"\n--- Bundle {idx + 1} ---")
        assert isinstance(bundle, dict), f"Bundle {idx} 应该是 dict"
        assert "splits" in bundle, f"Bundle {idx} 缺少 splits"
        assert "meta" in bundle, f"Bundle {idx} 缺少 meta"

        splits = bundle["splits"]
        meta = bundle["meta"]

        print(f"✓ Splits: {list(splits.keys())}")
        for split_name, split_data in splits.items():
            print(f"  - {split_name}: {len(split_data)} 样本")

        print(f"✓ Meta: {meta.get('dataset_name', 'N/A')}")
        print(f"  Total: {meta.get('total', 0)}")

    print("\n✓ 多数据集加载测试通过！\n")
    return True


def test_multi_datasets_yaml():
    """测试多数据集加载（使用 YAML 配置）"""
    print("=" * 80)
    print("测试 3: 多数据集加载（YAML 配置）")
    print("=" * 80)

    # 创建临时 YAML 配置
    import yaml
    import tempfile
    import os

    yaml_config = """
global_settings:
  seed: 42
  device: "cuda"
  dtype: "bfloat16"
  dataset_cache_dir: "/data/users/zjw/dataset_cache"
  hf_cache_dir: "/data/users/zjw/huggingface_cache"

dataset_settings:
  - type: "distill"
    name: "distill-openr1-math"
    distill_settings:
      split:
        train: 100
        test: -1
  - type: "qa"
    name: "qa-gsm8k"
    qa_settings:
      split:
        train: -1
        test: 50
"""

    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_config)
        temp_file = f.name

    try:
        print(f"\n加载临时配置文件: {temp_file}")
        config = load_config(override_file=temp_file)

        bundles = load_dataset(config)

        # 验证返回格式
        print(f"\n✓ 返回类型: {type(bundles)}")
        assert isinstance(bundles, list), "多数据集应返回 list"
        assert len(bundles) == 2, f"应返回 2 个 bundle，实际: {len(bundles)}"
        print(f"✓ 返回 {len(bundles)} 个 bundle")

        # 检查每个 bundle
        for idx, bundle in enumerate(bundles):
            print(f"\n--- Bundle {idx + 1} ---")
            splits = bundle["splits"]
            meta = bundle["meta"]

            print(f"✓ Splits: {list(splits.keys())}")
            for split_name, split_data in splits.items():
                print(f"  - {split_name}: {len(split_data)} 样本")

        print("\n✓ 多数据集 YAML 配置测试通过！\n")

    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return True


if __name__ == "__main__":
    try:
        # 测试 1: 单数据集（向后兼容）
        test_single_dataset()

        # 测试 2: 多数据集（dict）
        test_multi_datasets_dict()

        # 测试 3: 多数据集（YAML）
        test_multi_datasets_yaml()

        print("=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
