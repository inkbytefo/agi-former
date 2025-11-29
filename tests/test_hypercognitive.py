## Developer: inkbytefo
## Modified: 2025-11-30

import jax
import jax.numpy as jnp
from jax import random
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hypercognitive import hypercognitive_init, hypercognitive_apply, HyperCognitive

def test_hypercognitive_basic():
    """Test basic HyperCognitive functionality"""
    print("Testing basic HyperCognitive functionality...")

    # Parameters
    d_model = 128  # Smaller for testing
    num_branches = 3
    max_steps = 4
    batch_size = 2
    seq_len = 8

    key = random.PRNGKey(42)

    # Initialize
    params = hypercognitive_init(d_model, num_branches, max_steps, key)
    print(f"[OK] Initialized HyperCognitive with {num_branches} branches, max_steps={max_steps}")

    # Create test input
    x = random.normal(random.fold_in(key, 1), (batch_size, seq_len, d_model))
    thinking_steps = 3
    effort = 0.8

    # Apply - now returns (output, ortho_loss)
    result, ortho_loss = hypercognitive_apply(params, x, effort, train=True)
    print(f"[OK] Applied HyperCognitive: input shape {x.shape} -> output shape {result.shape}")

    # Check shape consistency
    assert result.shape == x.shape, "Output shape should match input shape"
    print("[OK] Shape consistency verified")

    # Check ortho_loss
    assert isinstance(ortho_loss, (float, jnp.ndarray)), "ortho_loss should be valid"
    print(f"[OK] Orthogonality loss: {ortho_loss}")

    # Check that result is different from input (transformation occurred)
    diff = jnp.mean(jnp.abs(result - x))
    assert diff > 0.01, "Output should be different from input"
    print(".3f")

    return True

def test_hypercognitive_class():
    """Test HyperCognitive class interface"""
    print("\nTesting HyperCognitive class interface...")

    d_model = 128
    num_branches = 3
    max_steps = 4

    # Create instance
    hc = HyperCognitive(d_model=d_model, num_branches=num_branches, max_steps=max_steps, key=random.PRNGKey(123))

    # Test forward - now returns (output, ortho_loss)
    batch_size, seq_len = 2, 8
    x = jnp.ones((batch_size, seq_len, d_model))
    result, ortho_loss = hc.forward(x, effort=0.7, train=True)

    assert result.shape == x.shape
    assert isinstance(ortho_loss, (float, jnp.ndarray))
    print(f"[OK] Class forward pass: shape {result.shape}, ortho_loss type {type(ortho_loss)}")

    return True

def test_diversity_and_synchronization():
    """Test branch diversity through embeddings"""
    print("\nTesting branch diversity...")

    d_model = 64
    num_branches = 4
    key = random.PRNGKey(100)

    # Create params
    params = hypercognitive_init(d_model, num_branches, 4, key)

    # Branch embeddings should be diverse
    embeddings = params["branch_embeddings"]  # (num_branches, d_model)
    for i in range(num_branches):
        for j in range(i+1, num_branches):
            diff = jnp.mean(jnp.abs(embeddings[i] - embeddings[j]))
            assert diff > 0.01, f"Embeddings {i} and {j} should be diverse"
            print(".3f")

    print("[OK] Branch embedding diversity verified")
    return True

def test_creative_merger():
    """Test creative merger through full hypercognitive"""
    print("\nTesting creative components (via full hypercognitive)...")

    d_model = 64
    num_branches = 3
    batch_size = 1
    seq_len = 6

    key = random.PRNGKey(200)

    # Test through full system
    params = hypercognitive_init(d_model, num_branches, 4, key)
    x = random.normal(key, (batch_size, seq_len, d_model))

    result, ortho_loss = hypercognitive_apply(params, x, effort=0.9, train=True)

    assert result.shape == x.shape
    assert isinstance(ortho_loss, (float, jnp.ndarray))
    print(f"[OK] Full hypercognitive with creative merger: shape {result.shape}")

    return True

if __name__ == "__main__":
    try:
        test_hypercognitive_basic()
        test_hypercognitive_class()
        test_diversity_and_synchronization()
        test_creative_merger()
        print("\n[SUCCESS] All tests passed!")
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
