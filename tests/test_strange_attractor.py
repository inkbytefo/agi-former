## Developer: inkbytefo
## Modified: 2025-11-30

import jax
import jax.numpy as jnp
from jax import random
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.strange_attractor import strange_attractor_init, strange_attractor_apply, StrangeAttractor

def test_strange_attractor_basic():
    """Test basic Strange Attractor functionality"""
    print("Testing basic Strange Attractor functionality...")

    # Parameters
    d_model = 128
    attractor_steps = 4
    batch_size = 1
    seq_len = 6

    key = random.PRNGKey(42)

    # Initialize
    params = strange_attractor_init(d_model, attractor_steps, key)
    print(f"[OK] Initialized Strange Attractor with steps={attractor_steps}")

    # Create test input
    x = random.normal(random.fold_in(key, 1), (batch_size, seq_len, d_model))

    # Apply - returns (output, stability_loss)
    result, stability_loss = strange_attractor_apply(params, x, train=True)
    print(f"[OK] Applied Strange Attractor: input shape {x.shape} -> output shape {result.shape}")

    # Check shape consistency
    assert result.shape == x.shape, "Output shape should match input shape"
    print("[OK] Shape consistency verified")

    # Check stability_loss
    assert isinstance(stability_loss, (float, jnp.ndarray)), "stability_loss should be valid"
    print(f"[OK] Stability loss: {stability_loss}")

    return True

def test_strange_attractor_class():
    """Test Strange Attractor class interface"""
    print("\nTesting Strange Attractor class interface...")

    d_model = 128
    attractor_steps = 4

    # Create instance
    sa = StrangeAttractor(d_model=d_model, attractor_steps=attractor_steps, key=random.PRNGKey(123))

    # Test forward
    batch_size, seq_len = 1, 6
    x = jnp.ones((batch_size, seq_len, d_model))
    result, stability_loss = sa.forward(x, train=True)

    assert result.shape == x.shape
    assert isinstance(stability_loss, (float, jnp.ndarray))
    print(f"[OK] Class forward pass: shape {result.shape}, stability_loss type {type(stability_loss)}")

    return True

def test_lyapunov_calculation():
    """Test Lyapunov exponent calculation in trajectory"""
    print("\nTesting Lyapunov exponent stability...")

    # Create a simple trajectory: exponential growth
    trajectory = jnp.array([
        [1.0, 1.0],
        [2.0, 1.5],
        [4.0, 2.25],
        [8.0, 3.375]
    ])  # (steps, D)

    from src.models.strange_attractor import compute_lyapunov_exponent
    lyap = compute_lyapunov_exponent(trajectory, None)

    assert lyap > 0, "Should detect chaotic growth"
    print(f"[OK] Lyapunov exponent for chaotic series: {lyap}")

    # Test with stable trajectory
    stable_traj = jnp.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
    ])
    lyap_stable = compute_lyapunov_exponent(stable_traj, None)
    assert lyap_stable <= 0, "Should detect stable behavior"
    print(f"[OK] Lyapunov exponent for stable series: {lyap_stable}")

    return True

if __name__ == "__main__":
    try:
        test_strange_attractor_basic()
        test_strange_attractor_class()
        test_lyapunov_calculation()
        print("\n[SUCCESS] All Strange Attractor tests passed!")
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
