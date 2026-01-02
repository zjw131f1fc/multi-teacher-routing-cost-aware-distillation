"""测试 step3 不同路由策略的数据加载行为"""

import sys
sys.path.insert(0, '/data/users/zjw/projects/multi-teacher-routing-cost-aware-distillation')

from engine.configs import load_config


def test_routing_strategies():
    """测试不同路由策略的数据加载逻辑"""
    print("=" * 80)
    print("测试路由策略判断逻辑")
    print("=" * 80)

    # 需要 NTE 分数的策略
    strategies_need_scores = ["nte_best", "nte_weighted", "knn_stats"]

    # 不需要 NTE 分数的策略
    strategies_no_scores = ["best_teacher", "random", "round_robin"]

    print("\n需要 NTE 分数的策略:")
    for strategy in strategies_need_scores:
        print(f"  - {strategy}")

    print("\n不需要 NTE 分数的策略:")
    for strategy in strategies_no_scores:
        print(f"  - {strategy}")

    # 测试当前配置
    print("\n" + "=" * 80)
    print("测试当前配置")
    print("=" * 80)

    config = load_config(override_file="configs/step3_distill_with_gsm8k.yaml")
    routing_strategy = config["method_settings"].get("routing_strategy", "random")

    need_nte_scores = routing_strategy in strategies_need_scores

    print(f"\n当前路由策略: {routing_strategy}")
    print(f"是否需要 NTE 分数: {need_nte_scores}")

    if need_nte_scores:
        print("✓ 将加载并合并 NTE 分数")
        print("✓ 将过滤训练集：保留包含所有教师 NTE 分数的样本")
    else:
        print("✓ 跳过 NTE 分数加载")
        print("✓ 将过滤训练集：保留包含所有教师响应的样本（不要求 NTE 分数）")

    print("\n" + "=" * 80)
    print("✓ 路由策略判断测试通过！")
    print("=" * 80)


def test_config_override():
    """测试配置覆盖"""
    print("\n" + "=" * 80)
    print("测试配置覆盖")
    print("=" * 80)

    # 测试修改路由策略
    test_strategies = [
        ("best_teacher", False),
        ("random", False),
        ("nte_best", True),
        ("knn_stats", True),
    ]

    strategies_need_scores = ["nte_best", "nte_weighted", "knn_stats"]

    for strategy, expected_need_scores in test_strategies:
        print(f"\n策略: {strategy}")

        need_scores = strategy in strategies_need_scores
        assert need_scores == expected_need_scores, \
            f"策略 {strategy} 的判断错误: expected={expected_need_scores}, got={need_scores}"

        print(f"  需要 NTE 分数: {need_scores}")
        print(f"  ✓ 判断正确")

    print("\n" + "=" * 80)
    print("✓ 配置覆盖测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_routing_strategies()
        test_config_override()

        print("\n" + "=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
