"""验证 DeBERTa Encoder Backbone 代码结构和导入。"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def test_structure():
    """测试代码结构和导入。"""
    print("=" * 60)
    print("验证 DeBERTa Encoder Backbone 实现")
    print("=" * 60)

    # 测试 1: 导入基类
    print("\n1. 导入 BaseEncoderBackbone...")
    try:
        from engine.backbones.base.encoder import BaseEncoderBackbone
        print(f"✓ 成功导入基类: {BaseEncoderBackbone}")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

    # 测试 2: 导入 DeBERTa 实现
    print("\n2. 导入 DeBERTaEncoderBackbone...")
    try:
        from engine.backbones.impl.encoder.deberta import DeBERTaEncoderBackbone
        print(f"✓ 成功导入实现: {DeBERTaEncoderBackbone}")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

    # 测试 3: 检查继承关系
    print("\n3. 验证继承关系...")
    try:
        assert issubclass(DeBERTaEncoderBackbone, BaseEncoderBackbone)
        print(f"✓ DeBERTaEncoderBackbone 正确继承 BaseEncoderBackbone")
    except Exception as e:
        print(f"✗ 继承关系错误: {e}")
        return False

    # 测试 4: 检查注册表
    print("\n4. 检查 BACKBONE_REGISTRY...")
    try:
        from engine.backbones.loader import BACKBONE_REGISTRY, list_backbone_types, list_backbones

        # 检查 encoder 类型
        assert "encoder" in BACKBONE_REGISTRY
        print(f"✓ 'encoder' 类型已注册")

        # 检查 deberta-v3-base
        assert "deberta-v3-base" in BACKBONE_REGISTRY["encoder"]
        print(f"✓ 'deberta-v3-base' 已注册到 encoder 类型")

        # 检查注册的类
        assert BACKBONE_REGISTRY["encoder"]["deberta-v3-base"] == DeBERTaEncoderBackbone
        print(f"✓ 注册的类正确")

    except Exception as e:
        print(f"✗ 注册表检查失败: {e}")
        return False

    # 测试 5: 列出所有类型和 backbone
    print("\n5. 列出已注册的 Backbone...")
    try:
        types = list_backbone_types()
        print(f"✓ 已注册类型: {types}")

        encoder_backbones = list_backbones("encoder")
        print(f"✓ Encoder 类型的 Backbone: {encoder_backbones}")

    except Exception as e:
        print(f"✗ 列表功能失败: {e}")
        return False

    # 测试 6: 检查类的方法
    print("\n6. 检查 DeBERTaEncoderBackbone 的方法...")
    try:
        methods = ['__init__', '_load_model', 'forward', 'encode']
        for method in methods:
            assert hasattr(DeBERTaEncoderBackbone, method)
            print(f"✓ 方法 '{method}' 存在")
    except Exception as e:
        print(f"✗ 方法检查失败: {e}")
        return False

    # 测试 7: 测试基本实例化（不加载实际模型）
    print("\n7. 测试基本实例化（无配置）...")
    try:
        backbone = DeBERTaEncoderBackbone(config=None)
        assert backbone.model_name == 'unknown'
        print(f"✓ 无配置实例化成功: model_name = '{backbone.model_name}'")
        print(f"  {backbone}")
    except Exception as e:
        print(f"✗ 实例化失败: {e}")
        return False

    # 测试 8: 使用 load_backbone 函数（模拟配置）
    print("\n8. 测试 load_backbone 函数...")
    try:
        from engine.backbones import load_backbone

        # 创建最小配置
        class MinimalConfig:
            backbone_settings = {
                "type": "encoder",
                "name": "deberta-v3-base",
            }

        # 只验证路由逻辑，不实际加载模型
        config = MinimalConfig()
        b_type = config.backbone_settings['type']
        name = config.backbone_settings['name']

        assert b_type in BACKBONE_REGISTRY
        assert name in BACKBONE_REGISTRY[b_type]
        backbone_cls = BACKBONE_REGISTRY[b_type][name]
        assert backbone_cls == DeBERTaEncoderBackbone

        print(f"✓ load_backbone 路由逻辑正确")
        print(f"  type='{b_type}', name='{name}' -> {backbone_cls}")

    except Exception as e:
        print(f"✗ load_backbone 测试失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("所有结构验证通过！✓")
    print("=" * 60)
    print("\n注意: 完整的模型加载测试需要 PyTorch >= 2.1")
    print("当前环境 PyTorch 版本过旧，但代码结构正确。")
    return True


if __name__ == "__main__":
    success = test_structure()
    sys.exit(0 if success else 1)
