"""测试 DeBERTa Encoder Backbone 实现。"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from engine.configs import load_config
from engine.backbones import load_backbone


def test_deberta_backbone():
    """测试 DeBERTa backbone 基本功能。"""
    print("=" * 60)
    print("测试 DeBERTa Encoder Backbone")
    print("=" * 60)

    # 加载真实配置
    print("\n1. 加载配置文件...")
    try:
        config = load_config(override_file="configs/test_deberta.yaml")
        print(f"✓ 配置加载成功")
        print(f"  Backbone type: {config.backbone_settings['type']}")
        print(f"  Backbone name: {config.backbone_settings['name']}")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 加载 backbone
    print("\n2. 加载 DeBERTa backbone...")
    try:
        backbone = load_backbone(config)
        print(f"✓ 成功加载: {backbone}")
        print(f"  Device: {backbone.device}")
        print(f"  Pooling mode: {backbone.pooling_mode}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试分词器
    print("\n3. 测试分词器...")
    try:
        test_text = "Hello, this is a test sentence."
        tokens = backbone.tokenizer(test_text, return_tensors="pt")
        print(f"✓ 分词成功: input_ids shape = {tokens['input_ids'].shape}")
        print(f"  Token IDs: {tokens['input_ids'][0][:10].tolist()}...")
    except Exception as e:
        print(f"✗ 分词失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 forward
    print("\n4. 测试 forward()...")
    try:
        outputs = backbone.forward(**tokens, return_dict=True)
        print(f"✓ forward 成功: last_hidden_state shape = {outputs.last_hidden_state.shape}")
        print(f"  Hidden size: {outputs.last_hidden_state.shape[-1]}")
    except Exception as e:
        print(f"✗ forward 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 encode (单文本)
    print("\n5. 测试 encode() - 单文本...")
    try:
        embedding = backbone.encode(test_text, pooling_mode="mean")
        print(f"✓ encode 成功: embedding shape = {embedding.shape}")
        print(f"  Hidden size: {embedding.shape[-1]}")
        print(f"  First 5 values: {embedding[:5].tolist()}")
    except Exception as e:
        print(f"✗ encode 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 encode (多文本)
    print("\n6. 测试 encode() - 多文本...")
    try:
        test_texts = [
            "This is the first sentence.",
            "This is the second sentence.",
            "And this is the third one.",
        ]
        embeddings = backbone.encode(test_texts, pooling_mode="mean")
        print(f"✓ encode 成功: embeddings shape = {embeddings.shape}")
        print(f"  Batch size: {embeddings.shape[0]}")
        print(f"  Hidden size: {embeddings.shape[1]}")
    except Exception as e:
        print(f"✗ encode 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试不同池化模式
    print("\n7. 测试不同池化模式...")
    try:
        for mode in ["cls", "mean", "max"]:
            emb = backbone.encode(test_text, pooling_mode=mode)
            print(f"✓ {mode} pooling: shape = {emb.shape}, norm = {emb.norm().item():.4f}")
    except Exception as e:
        print(f"✗ 池化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试返回 numpy
    print("\n8. 测试返回 numpy 数组...")
    try:
        embedding_np = backbone.encode(test_text, return_tensor=False)
        print(f"✓ 返回 numpy 成功")
        print(f"  Type: {type(embedding_np)}")
        print(f"  Shape: {embedding_np.shape}")
    except Exception as e:
        print(f"✗ numpy 返回失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_deberta_backbone()
    sys.exit(0 if success else 1)
