"""
Comprehensive tests for dual gradient backpropagation in fused-ssim.
Tests that gradients flow correctly to both img1 (predicted) and img2 (ground truth).
"""

import torch
import numpy as np
from fused_ssim import fused_ssim

# set tensor default to cuda
torch.set_default_tensor_type("torch.cuda.FloatTensor")


def test_img1_gradient_flow():
    """Test that gradients flow correctly to img1 (predicted image)."""
    print("Testing img1 gradient flow...")

    # Create test images
    torch.manual_seed(42)
    img1 = torch.rand(2, 3, 64, 64, requires_grad=True)
    img2 = torch.rand(2, 3, 64, 64, requires_grad=False)

    # Compute SSIM
    ssim_value = fused_ssim(img1, img2)

    # Backward pass
    ssim_value.backward()

    # Check that img1 has gradients
    assert img1.grad is not None, "img1 should have gradients"
    assert not torch.isnan(img1.grad).any(), "img1 gradients should not contain NaN"
    assert not torch.isinf(img1.grad).any(), "img1 gradients should not contain Inf"
    assert (img1.grad.abs() > 0).any(), "img1 gradients should be non-zero"

    print(f"✓ img1 gradient flow test passed")
    print(f"  SSIM value: {ssim_value.item():.6f}")
    print(f"  img1 grad mean: {img1.grad.mean().item():.6e}")
    print(f"  img1 grad std: {img1.grad.std().item():.6e}")
    print()


def test_img2_gradient_flow():
    """Test that gradients flow correctly to img2 (ground truth image)."""
    print("Testing img2 gradient flow...")

    # Create test images
    torch.manual_seed(42)
    img1 = torch.rand(2, 3, 64, 64, requires_grad=False)
    img2 = torch.rand(2, 3, 64, 64, requires_grad=True)

    # Compute SSIM
    ssim_value = fused_ssim(img1, img2)

    # Backward pass
    ssim_value.backward()

    # Check that img2 has gradients
    assert img2.grad is not None, "img2 should have gradients"
    assert not torch.isnan(img2.grad).any(), "img2 gradients should not contain NaN"
    assert not torch.isinf(img2.grad).any(), "img2 gradients should not contain Inf"
    assert (img2.grad.abs() > 0).any(), "img2 gradients should be non-zero"

    print(f"✓ img2 gradient flow test passed")
    print(f"  SSIM value: {ssim_value.item():.6f}")
    print(f"  img2 grad mean: {img2.grad.mean().item():.6e}")
    print(f"  img2 grad std: {img2.grad.std().item():.6e}")
    print()


def test_dual_gradient_flow():
    """Test that gradients flow correctly to both img1 and img2 simultaneously."""
    print("Testing dual gradient flow...")

    # Create test images - both require gradients
    torch.manual_seed(42)
    img1 = torch.rand(2, 3, 64, 64, requires_grad=True)
    img2 = torch.rand(2, 3, 64, 64, requires_grad=True)

    # Compute SSIM
    ssim_value = fused_ssim(img1, img2)

    # Backward pass
    ssim_value.backward()

    # Check that both have gradients
    assert img1.grad is not None, "img1 should have gradients"
    assert img2.grad is not None, "img2 should have gradients"

    assert not torch.isnan(img1.grad).any(), "img1 gradients should not contain NaN"
    assert not torch.isnan(img2.grad).any(), "img2 gradients should not contain NaN"

    assert not torch.isinf(img1.grad).any(), "img1 gradients should not contain Inf"
    assert not torch.isinf(img2.grad).any(), "img2 gradients should not contain Inf"

    assert (img1.grad.abs() > 0).any(), "img1 gradients should be non-zero"
    assert (img2.grad.abs() > 0).any(), "img2 gradients should be non-zero"

    print(f"✓ Dual gradient flow test passed")
    print(f"  SSIM value: {ssim_value.item():.6f}")
    print(
        f"  img1 grad mean: {img1.grad.mean().item():.6e}, std: {img1.grad.std().item():.6e}"
    )
    print(
        f"  img2 grad mean: {img2.grad.mean().item():.6e}, std: {img2.grad.std().item():.6e}"
    )
    print()


def test_gradient_symmetry():
    """Test that gradients are symmetric when images are swapped (SSIM is symmetric)."""
    print("Testing gradient symmetry...")

    torch.manual_seed(42)
    img_a = torch.rand(1, 3, 32, 32, requires_grad=True)
    img_b = torch.rand(1, 3, 32, 32, requires_grad=True)

    # Compute SSIM(img_a, img_b)
    ssim_ab = fused_ssim(img_a, img_b)
    ssim_ab.backward()
    grad_a_from_ab = img_a.grad.clone()
    grad_b_from_ab = img_b.grad.clone()

    # Reset gradients
    img_a.grad = None
    img_b.grad = None

    # Compute SSIM(img_b, img_a) - swapped
    ssim_ba = fused_ssim(img_b, img_a)
    ssim_ba.backward()
    grad_b_from_ba = img_b.grad.clone()
    grad_a_from_ba = img_a.grad.clone()

    # SSIM values should be identical
    assert torch.allclose(
        ssim_ab, ssim_ba, rtol=1e-5, atol=1e-7
    ), "SSIM should be symmetric: SSIM(A,B) == SSIM(B,A)"

    # Gradients should be swapped: grad_a from (A,B) should match grad_a from (B,A)
    # and grad_b from (A,B) should match grad_b from (B,A)
    assert torch.allclose(
        grad_a_from_ab, grad_a_from_ba, rtol=1e-4, atol=1e-6
    ), "Gradient of img_a should be consistent regardless of argument order"
    assert torch.allclose(
        grad_b_from_ab, grad_b_from_ba, rtol=1e-4, atol=1e-6
    ), "Gradient of img_b should be consistent regardless of argument order"

    print(f"✓ Gradient symmetry test passed")
    print(f"  SSIM(A,B): {ssim_ab.item():.6f}, SSIM(B,A): {ssim_ba.item():.6f}")
    print()


def test_numerical_gradient_img1():
    """Verify img1 gradients using numerical gradient checking."""
    print("Testing numerical gradient for img1...")

    torch.manual_seed(42)
    img1 = torch.rand(1, 1, 16, 16, requires_grad=True)
    img2 = torch.rand(1, 1, 16, 16, requires_grad=False)

    # Compute analytical gradient
    ssim_value = fused_ssim(img1, img2)
    ssim_value.backward()
    analytical_grad = img1.grad.clone()

    # Compute numerical gradient using finite differences
    epsilon = 1e-4
    numerical_grad = torch.zeros_like(img1)

    # Sample a few random positions for efficiency
    np.random.seed(42)
    num_samples = 20
    for _ in range(num_samples):
        b, c, h, w = 0, 0, np.random.randint(0, 16), np.random.randint(0, 16)

        # Forward difference
        img1.data[b, c, h, w] += epsilon
        ssim_plus = fused_ssim(img1, img2)

        img1.data[b, c, h, w] -= 2 * epsilon
        ssim_minus = fused_ssim(img1, img2)

        img1.data[b, c, h, w] += epsilon  # restore

        numerical_grad[b, c, h, w] = (ssim_plus - ssim_minus) / (2 * epsilon)

    # Compare analytical and numerical gradients at sampled positions
    mask = numerical_grad != 0
    relative_error = torch.abs(analytical_grad[mask] - numerical_grad[mask]) / (
        torch.abs(analytical_grad[mask]) + torch.abs(numerical_grad[mask]) + 1e-8
    )

    max_error = relative_error.max().item()
    mean_error = relative_error.mean().item()

    print(f"✓ Numerical gradient test for img1 passed")
    print(f"  Max relative error: {max_error:.6e}")
    print(f"  Mean relative error: {mean_error:.6e}")
    assert (
        max_error < 0.1
    ), f"Gradient error too large: {max_error}"  # 10% tolerance for finite differences
    print()


def test_numerical_gradient_img2():
    """Verify img2 gradients using numerical gradient checking."""
    print("Testing numerical gradient for img2...")

    torch.manual_seed(42)
    img1 = torch.rand(1, 1, 16, 16, requires_grad=False)
    img2 = torch.rand(1, 1, 16, 16, requires_grad=True)

    # Compute analytical gradient
    ssim_value = fused_ssim(img1, img2)
    ssim_value.backward()
    analytical_grad = img2.grad.clone()

    # Compute numerical gradient using finite differences
    epsilon = 1e-4
    numerical_grad = torch.zeros_like(img2)

    # Sample a few random positions for efficiency
    np.random.seed(42)
    num_samples = 20
    for _ in range(num_samples):
        b, c, h, w = 0, 0, np.random.randint(0, 16), np.random.randint(0, 16)

        # Forward difference
        img2.data[b, c, h, w] += epsilon
        ssim_plus = fused_ssim(img1, img2)

        img2.data[b, c, h, w] -= 2 * epsilon
        ssim_minus = fused_ssim(img1, img2)

        img2.data[b, c, h, w] += epsilon  # restore

        numerical_grad[b, c, h, w] = (ssim_plus - ssim_minus) / (2 * epsilon)

    # Compare analytical and numerical gradients at sampled positions
    mask = numerical_grad != 0
    relative_error = torch.abs(analytical_grad[mask] - numerical_grad[mask]) / (
        torch.abs(analytical_grad[mask]) + torch.abs(numerical_grad[mask]) + 1e-8
    )

    max_error = relative_error.max().item()
    mean_error = relative_error.mean().item()

    print(f"✓ Numerical gradient test for img2 passed")
    print(f"  Max relative error: {max_error:.6e}")
    print(f"  Mean relative error: {mean_error:.6e}")
    assert (
        max_error < 0.1
    ), f"Gradient error too large: {max_error}"  # 10% tolerance for finite differences
    print()


def test_gradient_descent_img1():
    """Test that gradient descent on img1 improves SSIM."""
    print("Testing gradient descent on img1...")

    torch.manual_seed(42)
    img1 = torch.rand(1, 3, 32, 32, requires_grad=True)
    img2 = torch.rand(1, 3, 32, 32, requires_grad=False)

    # Initial SSIM
    initial_ssim = fused_ssim(img1, img2).item()

    # Perform gradient ascent to maximize SSIM
    optimizer = torch.optim.Adam([img1], lr=0.01)
    for _ in range(10):
        optimizer.zero_grad()
        ssim = fused_ssim(img1, img2)
        loss = -ssim  # Negative because we want to maximize SSIM
        loss.backward()
        optimizer.step()
        img1.data.clamp_(0, 1)  # Keep values in valid range

    final_ssim = fused_ssim(img1, img2).item()

    print(f"✓ Gradient descent test for img1 passed")
    print(f"  Initial SSIM: {initial_ssim:.6f}")
    print(f"  Final SSIM: {final_ssim:.6f}")
    print(f"  Improvement: {final_ssim - initial_ssim:.6f}")
    assert final_ssim > initial_ssim, "SSIM should improve with gradient ascent"
    print()


def test_gradient_descent_img2():
    """Test that gradient descent on img2 improves SSIM."""
    print("Testing gradient descent on img2...")

    torch.manual_seed(42)
    img1 = torch.rand(1, 3, 32, 32, requires_grad=False)
    img2 = torch.rand(1, 3, 32, 32, requires_grad=True)

    # Initial SSIM
    initial_ssim = fused_ssim(img1, img2).item()

    # Perform gradient ascent to maximize SSIM
    optimizer = torch.optim.Adam([img2], lr=0.01)
    for _ in range(10):
        optimizer.zero_grad()
        ssim = fused_ssim(img1, img2)
        loss = -ssim  # Negative because we want to maximize SSIM
        loss.backward()
        optimizer.step()
        img2.data.clamp_(0, 1)  # Keep values in valid range

    final_ssim = fused_ssim(img1, img2).item()

    print(f"✓ Gradient descent test for img2 passed")
    print(f"  Initial SSIM: {initial_ssim:.6f}")
    print(f"  Final SSIM: {final_ssim:.6f}")
    print(f"  Improvement: {final_ssim - initial_ssim:.6f}")
    assert final_ssim > initial_ssim, "SSIM should improve with gradient ascent"
    print()


def test_valid_padding():
    """Test dual gradients with valid padding mode."""
    print("Testing dual gradients with valid padding...")

    torch.manual_seed(42)
    img1 = torch.rand(1, 3, 32, 32, requires_grad=True)
    img2 = torch.rand(1, 3, 32, 32, requires_grad=True)

    # Compute SSIM with valid padding
    ssim_value = fused_ssim(img1, img2, padding="valid")
    ssim_value.backward()

    # Check that both have gradients
    assert img1.grad is not None, "img1 should have gradients"
    assert img2.grad is not None, "img2 should have gradients"
    assert not torch.isnan(img1.grad).any(), "img1 gradients should not contain NaN"
    assert not torch.isnan(img2.grad).any(), "img2 gradients should not contain NaN"

    print(f"✓ Valid padding test passed")
    print(f"  SSIM value: {ssim_value.item():.6f}")
    print()


def test_different_batch_sizes():
    """Test dual gradients with various batch sizes."""
    print("Testing dual gradients with different batch sizes...")

    for batch_size in [1, 2, 4, 8]:
        torch.manual_seed(42)
        img1 = torch.rand(batch_size, 3, 32, 32, requires_grad=True)
        img2 = torch.rand(batch_size, 3, 32, 32, requires_grad=True)

        ssim_value = fused_ssim(img1, img2)
        ssim_value.backward()

        assert (
            img1.grad is not None
        ), f"img1 should have gradients for batch_size={batch_size}"
        assert (
            img2.grad is not None
        ), f"img2 should have gradients for batch_size={batch_size}"
        assert not torch.isnan(
            img1.grad
        ).any(), f"img1 gradients should be valid for batch_size={batch_size}"
        assert not torch.isnan(
            img2.grad
        ).any(), f"img2 gradients should be valid for batch_size={batch_size}"

        print(f"  ✓ Batch size {batch_size}: SSIM={ssim_value.item():.6f}")

    print(f"✓ Different batch sizes test passed")
    print()


def test_different_image_sizes():
    """Test dual gradients with various image sizes."""
    print("Testing dual gradients with different image sizes...")

    for size in [16, 32, 64, 128]:
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, size, size, requires_grad=True)
        img2 = torch.rand(1, 3, size, size, requires_grad=True)

        ssim_value = fused_ssim(img1, img2)
        ssim_value.backward()

        assert img1.grad is not None, f"img1 should have gradients for size={size}"
        assert img2.grad is not None, f"img2 should have gradients for size={size}"
        assert not torch.isnan(
            img1.grad
        ).any(), f"img1 gradients should be valid for size={size}"
        assert not torch.isnan(
            img2.grad
        ).any(), f"img2 gradients should be valid for size={size}"

        print(f"  ✓ Image size {size}x{size}: SSIM={ssim_value.item():.6f}")

    print(f"✓ Different image sizes test passed")
    print()


def run_all_tests():
    """Run all comprehensive tests."""
    print("=" * 70)
    print("COMPREHENSIVE DUAL GRADIENT TESTS FOR FUSED-SSIM")
    print("=" * 70)
    print()

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Running tests on CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.mps.is_available():
        device = "mps"
        print(f"Running tests on Apple Metal (MPS)")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
        print(f"Running tests on Intel XPU")
    else:
        print("ERROR: No GPU device available!")
        return

    print()

    tests = [
        test_img1_gradient_flow,
        test_img2_gradient_flow,
        test_dual_gradient_flow,
        test_gradient_symmetry,
        test_numerical_gradient_img1,
        test_numerical_gradient_img2,
        test_gradient_descent_img1,
        test_gradient_descent_img2,
        test_valid_padding,
        test_different_batch_sizes,
        test_different_image_sizes,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {str(e)}")
            print()
            failed += 1

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    if failed == 0:
        print("🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
