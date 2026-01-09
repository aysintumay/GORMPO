"""
Test script to verify the StandardNormalizer.update() method fix.
Tests both numpy arrays and torch tensors.
"""

import numpy as np
import torch
import sys
sys.path.append('/home/ubuntu/GORMPO')
from common.normalizer import StandardNormalizer


def test_update_with_numpy():
    """Test update method with numpy arrays"""
    print("=" * 60)
    print("TEST 1: Update with NumPy arrays")
    print("=" * 60)

    normalizer = StandardNormalizer()

    # First batch
    batch1 = np.random.randn(100, 5).astype(np.float32)
    print(f"Batch 1 shape: {batch1.shape}, type: {type(batch1)}")

    try:
        normalizer.update(batch1)
        print(f"✓ First update successful")
        print(f"  Mean shape: {normalizer.mean.shape}, type: {type(normalizer.mean)}")
        print(f"  Var shape: {normalizer.var.shape}, type: {type(normalizer.var)}")
        print(f"  Tot count: {normalizer.tot_count}")
    except Exception as e:
        print(f"✗ First update failed: {e}")
        return False

    # Second batch
    batch2 = np.random.randn(50, 5).astype(np.float32)
    print(f"\nBatch 2 shape: {batch2.shape}, type: {type(batch2)}")

    try:
        normalizer.update(batch2)
        print(f"✓ Second update successful")
        print(f"  Mean shape: {normalizer.mean.shape}, type: {type(normalizer.mean)}")
        print(f"  Var shape: {normalizer.var.shape}, type: {type(normalizer.var)}")
        print(f"  Tot count: {normalizer.tot_count}")
    except Exception as e:
        print(f"✗ Second update failed: {e}")
        return False

    # Test transform
    test_data = np.random.randn(10, 5).astype(np.float32)
    try:
        transformed = normalizer.transform(test_data)
        print(f"\n✓ Transform successful")
        print(f"  Transformed shape: {transformed.shape}, type: {type(transformed)}")
    except Exception as e:
        print(f"\n✗ Transform failed: {e}")
        return False

    print("\n✓ NumPy test PASSED\n")
    return True


def test_update_with_torch():
    """Test update method with torch tensors"""
    print("=" * 60)
    print("TEST 2: Update with Torch tensors")
    print("=" * 60)

    normalizer = StandardNormalizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # First batch
    batch1 = torch.randn(100, 5, device=device)
    print(f"Batch 1 shape: {batch1.shape}, type: {type(batch1)}, device: {batch1.device}")

    try:
        normalizer.update(batch1)
        print(f"✓ First update successful")
        print(f"  Mean shape: {normalizer.mean.shape}, type: {type(normalizer.mean)}")
        print(f"  Var shape: {normalizer.var.shape}, type: {type(normalizer.var)}")
        print(f"  Tot count: {normalizer.tot_count}")
        if torch.is_tensor(normalizer.mean):
            print(f"  Mean device: {normalizer.mean.device}")
    except Exception as e:
        print(f"✗ First update failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Second batch
    batch2 = torch.randn(50, 5, device=device)
    print(f"\nBatch 2 shape: {batch2.shape}, type: {type(batch2)}, device: {batch2.device}")

    try:
        normalizer.update(batch2)
        print(f"✓ Second update successful")
        print(f"  Mean shape: {normalizer.mean.shape}, type: {type(normalizer.mean)}")
        print(f"  Var shape: {normalizer.var.shape}, type: {type(normalizer.var)}")
        print(f"  Tot count: {normalizer.tot_count}")
    except Exception as e:
        print(f"✗ Second update failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test transform
    test_data = torch.randn(10, 5, device=device)
    try:
        transformed = normalizer.transform(test_data)
        print(f"\n✓ Transform successful")
        print(f"  Transformed shape: {transformed.shape}, type: {type(transformed)}")
    except Exception as e:
        print(f"\n✗ Transform failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ Torch test PASSED\n")
    return True


def test_update_statistics():
    """Test that update method computes correct statistics"""
    print("=" * 60)
    print("TEST 3: Statistical correctness")
    print("=" * 60)

    normalizer = StandardNormalizer()

    # Create two batches with varying data
    np.random.seed(42)
    batch1 = np.random.randn(100, 3).astype(np.float32)
    batch2 = np.random.randn(100, 3).astype(np.float32)

    # Compute expected statistics using all data at once
    all_data = np.concatenate([batch1, batch2], axis=0)
    expected_mean = np.mean(all_data, axis=0)
    expected_var = np.var(all_data, axis=0)
    expected_var[expected_var < 1e-12] = 1.0  # Apply same clamping

    # Update incrementally
    normalizer.update(batch1)
    print(f"After batch1:")
    print(f"  Mean: {normalizer.mean[0, :3]}")
    print(f"  Var: {normalizer.var[0, :3]}")

    normalizer.update(batch2)
    print(f"\nAfter batch2:")
    print(f"  Mean: {normalizer.mean[0, :3]}")
    print(f"  Var: {normalizer.var[0, :3]}")
    print(f"\nExpected (computed from all data):")
    print(f"  Mean: {expected_mean}")
    print(f"  Var: {expected_var}")

    # Check if close to expected values
    mean_correct = np.allclose(normalizer.mean[0], expected_mean, rtol=1e-4, atol=1e-6)
    var_correct = np.allclose(normalizer.var[0], expected_var, rtol=1e-4, atol=1e-6)

    if mean_correct and var_correct:
        print("\n✓ Statistical correctness PASSED")
        print("  (Incremental updates match batch computation)")
    else:
        print("\n✗ Statistical correctness FAILED")
        print(f"  Mean match: {mean_correct}")
        print(f"  Var match: {var_correct}")
        print(f"  Mean error: {np.abs(normalizer.mean[0] - expected_mean).max()}")
        print(f"  Var error: {np.abs(normalizer.var[0] - expected_var).max()}")

    print()
    return mean_correct and var_correct


def test_mixed_types():
    """Test updating with numpy then torch (or vice versa)"""
    print("=" * 60)
    print("TEST 4: Mixed types (NumPy then Torch)")
    print("=" * 60)

    normalizer = StandardNormalizer()

    # Start with numpy
    batch1 = np.random.randn(100, 5).astype(np.float32)
    print(f"Batch 1 (NumPy) shape: {batch1.shape}")

    try:
        normalizer.update(batch1)
        print(f"✓ NumPy update successful")
    except Exception as e:
        print(f"✗ NumPy update failed: {e}")
        return False

    # Follow with torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch2 = torch.randn(50, 5, device=device)
    print(f"Batch 2 (Torch) shape: {batch2.shape}, device: {batch2.device}")

    try:
        normalizer.update(batch2)
        print(f"✓ Torch update successful")
        print(f"  Total count: {normalizer.tot_count}")
    except Exception as e:
        print(f"✗ Torch update failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ Mixed types test PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING STANDARDNORMALIZER UPDATE FIX")
    print("=" * 60 + "\n")

    results = []

    # Run all tests
    results.append(("NumPy arrays", test_update_with_numpy()))
    results.append(("Torch tensors", test_update_with_torch()))
    results.append(("Statistical correctness", test_update_statistics()))
    results.append(("Mixed types", test_mixed_types()))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30} {status}")

    all_passed = all(result[1] for result in results)
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
