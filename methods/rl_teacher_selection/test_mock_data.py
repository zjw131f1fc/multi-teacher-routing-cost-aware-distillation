"""Test mock data loading"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from methods.rl_teacher_selection import get_mock_data


def main():
    print("=" * 60)
    print("Testing Mock Data Loading")
    print("=" * 60)

    # Load all mock data
    print("\n[1] Loading mock data...")
    data = get_mock_data(
        num_instructions=100,
        num_val_samples=50,
        device="cpu"
    )
    print("   ✓ Loaded")

    # Check instructions
    print("\n[2] Instructions:")
    print(f"   Count: {len(data['instructions'])}")
    print(f"   Examples:")
    for i in range(3):
        print(f"     [{i}] {data['instructions'][i]}")

    # Check student model
    print("\n[3] Student Model:")
    num_params = sum(p.numel() for p in data['student_model'].parameters())
    print(f"   Parameters: {num_params:,}")
    print(f"   Device: {next(data['student_model'].parameters()).device}")

    # Check teachers
    print("\n[4] Teachers:")
    print(f"   Count: {len(data['teachers'])}")
    for teacher in data['teachers']:
        print(f"   - {teacher.name}: quality={teacher.quality}, cost={teacher.cost}")

    # Check validation loader
    print("\n[5] Validation Loader:")
    print(f"   Batches: {len(data['val_loader'])}")
    batch = next(iter(data['val_loader']))
    print(f"   Batch shape: input_ids={batch[0].shape}, labels={batch[1].shape}")

    # Test student model forward
    print("\n[6] Testing student model forward pass...")
    input_ids, labels = batch
    loss = data['student_model'](input_ids, labels)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✓ Forward pass works")

    # Test teacher generation
    print("\n[7] Testing teacher generation...")
    teacher = data['teachers'][0]
    instruction = data['instructions'][0]
    gen_input_ids, gen_labels = teacher.generate(instruction)
    print(f"   Teacher: {teacher.name}")
    print(f"   Instruction: {instruction}")
    print(f"   Generated length: {len(gen_input_ids)}")
    print(f"   ✓ Generation works")

    # Test cache mechanism
    print("\n[8] Testing cache mechanism...")
    teacher = data['teachers'][0]

    # First call - cache miss
    instruction = data['instructions'][0]
    result1 = teacher.generate(instruction)
    stats1 = teacher.get_cache_stats()
    print(f"   After 1st call: cache_size={stats1['cache_size']}, hits={stats1['cache_hits']}, misses={stats1['cache_misses']}")

    # Second call - cache hit
    result2 = teacher.generate(instruction)
    stats2 = teacher.get_cache_stats()
    print(f"   After 2nd call: cache_size={stats2['cache_size']}, hits={stats2['cache_hits']}, misses={stats2['cache_misses']}")

    # Verify same result
    assert torch.equal(result1[0], result2[0]), "Cache should return same result"
    print(f"   Hit rate: {stats2['hit_rate']:.1%}")
    print(f"   ✓ Cache works correctly")

    # Test prefill
    print("\n[9] Testing cache prefill...")
    teacher2 = data['teachers'][1]
    teacher2.prefill_cache(data['instructions'][:10])
    stats = teacher2.get_cache_stats()
    print(f"   Prefilled {stats['cache_size']} instructions")
    print(f"   ✓ Prefill works")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
