"""Test script for PolicyNetwork"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from methods.rl_teacher_selection import PolicyNetwork


def test_policy_network():
    """Test PolicyNetwork basic functionality."""
    print("=" * 60)
    print("Testing PolicyNetwork")
    print("=" * 60)

    # Configuration
    num_teachers = 3
    num_actions = num_teachers + 1  # +1 for Reuse
    hidden_dims = [512, 256]

    print(f"\nConfiguration:")
    print(f"  Num teachers: {num_teachers}")
    print(f"  Num actions: {num_actions} (T1, T2, T3, Reuse)")
    print(f"  Hidden dims: {hidden_dims}")

    # Initialize policy
    print("\n[1] Initializing PolicyNetwork...")
    policy = PolicyNetwork(
        num_actions=num_actions,
        encoder_name="microsoft/deberta-v3-base",
        hidden_dims=hidden_dims,
        freeze_encoder=True,
    )
    print(f"  ✓ Policy initialized")
    print(f"  ✓ Encoder: {policy.encoder_name}")
    print(f"  ✓ Encoder frozen: {policy.freeze_encoder}")

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n[2] Parameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {frozen_params:,}")

    # Test single instruction
    print("\n[3] Testing single instruction...")
    instruction = "Solve the equation: 2x + 5 = 13"
    probs = policy(instruction)

    print(f"  Input: '{instruction}'")
    print(f"  Output shape: {probs.shape}")
    print(f"  Probabilities: {probs[0].tolist()}")
    print(f"  Sum of probs: {probs.sum().item():.6f}")
    assert probs.shape == (1, num_actions), f"Expected shape (1, {num_actions}), got {probs.shape}"
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6), "Probabilities should sum to 1"
    print(f"  ✓ Single instruction test passed")

    # Test batch of instructions
    print("\n[4] Testing batch of instructions...")
    instructions = [
        "What is the capital of France?",
        "Calculate 15 * 23",
        "Explain photosynthesis",
    ]
    probs_batch = policy(instructions)

    print(f"  Batch size: {len(instructions)}")
    print(f"  Output shape: {probs_batch.shape}")
    print(f"  Probabilities:")
    for i, inst in enumerate(instructions):
        print(f"    [{i}] {probs_batch[i].tolist()}")
    print(f"  Sum of probs per sample: {probs_batch.sum(dim=1).tolist()}")
    assert probs_batch.shape == (len(instructions), num_actions)
    assert torch.allclose(probs_batch.sum(dim=1), torch.ones(len(instructions)), atol=1e-6)
    print(f"  ✓ Batch test passed")

    # Test action sampling
    print("\n[5] Testing action sampling...")
    actions, log_probs = policy.sample_action(instructions)

    print(f"  Sampled actions: {actions.tolist()}")
    print(f"  Log probabilities: {log_probs.tolist()}")
    assert actions.shape == (len(instructions),)
    assert log_probs.shape == (len(instructions),)
    assert all(0 <= a < num_actions for a in actions.tolist())
    print(f"  ✓ Action sampling test passed")

    # Test get_action_probs
    print("\n[6] Testing get_action_probs...")
    specific_actions = torch.tensor([0, 1, 3])  # T1, T2, Reuse
    action_probs = policy.get_action_probs(instructions, specific_actions)

    print(f"  Specific actions: {specific_actions.tolist()}")
    print(f"  Action probabilities: {action_probs.tolist()}")
    assert action_probs.shape == (len(instructions),)
    print(f"  ✓ get_action_probs test passed")

    # Test gradient flow
    print("\n[7] Testing gradient flow...")
    probs = policy(instruction)
    loss = -probs[0, 0]  # Dummy loss
    loss.backward()

    encoder_has_grad = any(p.grad is not None for p in policy.encoder.parameters())
    mlp_has_grad = any(p.grad is not None for p in policy.mlp_head.parameters())

    print(f"  Encoder has gradients: {encoder_has_grad}")
    print(f"  MLP head has gradients: {mlp_has_grad}")
    assert not encoder_has_grad, "Encoder should be frozen"
    assert mlp_has_grad, "MLP head should have gradients"
    print(f"  ✓ Gradient flow test passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_policy_network()
