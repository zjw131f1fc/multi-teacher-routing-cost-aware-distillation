"""Simple test for PolicyNetwork"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModel, AutoTokenizer
from methods.rl_teacher_selection import PolicyNetwork


def main():
    print("Testing PolicyNetwork...")

    # Load encoder and tokenizer
    print("\n1. Loading DeBERTa-v3-base...")
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    print("   ✓ Loaded")

    # Create policy
    print("\n2. Creating PolicyNetwork...")
    num_actions = 4  # 3 teachers + 1 reuse
    policy = PolicyNetwork(
        encoder=encoder,
        tokenizer=tokenizer,
        num_actions=num_actions,
        hidden_dims=[512, 256],
        freeze_encoder=True,
    )
    print(f"   ✓ Created with {num_actions} actions")

    # Count parameters
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    print(f"   Trainable params: {trainable:,}")
    print(f"   Frozen params: {frozen:,}")

    # Test forward
    print("\n3. Testing forward pass...")
    instruction = "Solve: 2x + 5 = 13"
    probs = policy(instruction)
    print(f"   Input: '{instruction}'")
    print(f"   Output shape: {probs.shape}")
    print(f"   Probabilities: {probs[0].detach().numpy()}")
    print(f"   Sum: {probs.sum().item():.6f}")

    # Test batch
    print("\n4. Testing batch...")
    instructions = ["Question 1", "Question 2", "Question 3"]
    probs = policy(instructions)
    print(f"   Batch size: {len(instructions)}")
    print(f"   Output shape: {probs.shape}")

    # Test sampling
    print("\n5. Testing action sampling...")
    actions, log_probs = policy.sample_action(instructions)
    print(f"   Actions: {actions.tolist()}")
    print(f"   Log probs: {log_probs.tolist()}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
