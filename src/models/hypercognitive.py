## Developer: inkbytefo
## Modified: 2025-11-30

"""
HyperCognitive Module: Advanced Reasoning Capabilities for AGIFormer
Optimized for JAX/Flax with Orthogonality Constraints.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Dict, Any, Tuple


def hypercognitive_init(d_model: int, num_branches: int, max_steps: int, key: random.PRNGKey) -> Dict[str, Any]:
    """
    Initialize HyperCognitive module components.
    """
    k1, k2, k3, k4, k5 = random.split(key, 5)

    return {
        # 1. Diversity & Expansion (Divergent Thinking)
        # Her branch için farklı bir semantik başlangıç noktası öğrenir
        "branch_embeddings": random.normal(k1, (num_branches, d_model)) * 0.02,

        # 2. Reasoning Branches (Parallel Processing)
        "branches": [reasoning_branch_init(d_model, random.fold_in(k2, i)) for i in range(num_branches)],

        # 3. Semantic Synchronizer (Lateral Communication)
        "synchronizer": attention_init(d_model, k3),

        # 4. Creative Merger (Convergent Synthesis)
        "merger": creative_merger_init(d_model, num_branches, k4),

        # 5. Output Projection
        "output_norm": init_layer_norm(d_model),
        "output_proj": random.normal(k5, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model)),

        # Configs
        "num_branches": num_branches,
        "max_steps": max_steps,
        "d_model": d_model
    }


def reasoning_branch_init(d_model: int, key: random.PRNGKey) -> Dict[str, Any]:
    """Initialize parameters for a single thinking branch."""
    k1, k2 = random.split(key, 2)
    return {
        "W1": random.normal(k1, (d_model, d_model * 2)) * (1.0 / jnp.sqrt(d_model)), # Expansion
        "b1": jnp.zeros((d_model * 2,)),
        "W2": random.normal(k2, (d_model * 2, d_model)) * (1.0 / jnp.sqrt(d_model * 2)), # Compression
        "b2": jnp.zeros((d_model,)),
        "ln_gamma": jnp.ones((d_model,)),
        "ln_beta": jnp.zeros((d_model,))
    }

def attention_init(d_model: int, key: random.PRNGKey) -> Dict[str, Any]:
    """Standard Multi-Head Attention initialization."""
    k_q, k_k, k_v, k_o = random.split(key, 4)
    return {
        "W_q": random.normal(k_q, (d_model, d_model)) * (d_model ** -0.5),
        "W_k": random.normal(k_k, (d_model, d_model)) * (d_model ** -0.5),
        "W_v": random.normal(k_v, (d_model, d_model)) * (d_model ** -0.5),
        "W_o": random.normal(k_o, (d_model, d_model)) * (d_model ** -0.5),
    }

def creative_merger_init(d_model: int, num_branches: int, key: random.PRNGKey) -> Dict[str, Any]:
    """Merger specifically designed to find novel combinations."""
    k1, k2 = random.split(key, 2)
    return {
        # Attention to weigh branch importance based on context
        "query_proj": random.normal(k1, (d_model, d_model)) * (d_model ** -0.5),
        "key_proj": random.normal(k2, (d_model, d_model)) * (d_model ** -0.5),
    }

def init_layer_norm(d_model: int):
    return {"gamma": jnp.ones((d_model,)), "beta": jnp.zeros((d_model,))}

# --- Runtime Functions ---

def layer_norm(x: jnp.ndarray, params):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return params["gamma"] * (x - mean) / jnp.sqrt(var + 1e-6) + params["beta"]

def hypercognitive_apply(params: Dict[str, Any], x: jnp.ndarray, effort: float, train: bool = True) -> Tuple[jnp.ndarray, float]:
    """
    Main execution logic.
    Returns: (output_state, orthogonality_loss)
    """
    B, L, D = x.shape
    num_branches = params["num_branches"]

    # 1. DIVERGENCE: Branch Expansion
    # Her branch'e orijinal input + o branch'e özel öğrenilebilir bir "kişilik" (embedding) eklenir.
    # Bu, branch 1'in "mantıksal", branch 2'nin "duygusal" vb. özelleşmesini sağlar.
    branch_inputs = []
    for i in range(num_branches):
        # Broadcast embedding: (1, 1, D) -> (B, L, D)
        bias = params["branch_embeddings"][i][None, None, :]
        branch_inputs.append(x + bias)

    # Stack: (B, L, num_branches, D)
    current_thoughts = jnp.stack(branch_inputs, axis=2)

    # 2. PARALLEL REASONING PROCESS
    # Basit bir loop yerine, effort seviyesine göre adım sayısı belirlenir.
    steps = int(params["max_steps"] * effort) + 1

    def branch_process(state, _):
        # state shape: (B, L, num_branches, D)
        outputs = []
        for i in range(num_branches):
            b_params = params["branches"][i]
            inp = state[:, :, i, :]

            # MLP Block with Residual
            normed = layer_norm(inp, {"gamma": b_params["ln_gamma"], "beta": b_params["ln_beta"]})
            hidden = jax.nn.gelu(jnp.dot(normed, b_params["W1"]) + b_params["b1"])
            out = jnp.dot(hidden, b_params["W2"]) + b_params["b2"]
            outputs.append(inp + out) # Residual update

        return jnp.stack(outputs, axis=2), None

    # Iterative thinking
    final_thoughts, _ = lax.scan(branch_process, current_thoughts, None, length=steps)

    # 3. SEMANTIC SYNCHRONIZATION (Cross-Talk)
    # Branch'ler birbirinin ne düşündüğünden haberdar olur ama kaynaşmazlar.
    # Flatten branches for attention: Query=Thoughts, Key/Val=Thoughts
    # Basitleştirilmiş self-attention over branches dimension
    flat_thoughts = final_thoughts.reshape(B * L, num_branches, D)
    sync_params = params["synchronizer"]

    # Simple Attention mechanism
    Q = jnp.dot(flat_thoughts, sync_params["W_q"])
    K = jnp.dot(flat_thoughts, sync_params["W_k"])
    V = jnp.dot(flat_thoughts, sync_params["W_v"])

    logits = jnp.matmul(Q, K.transpose(0, 2, 1)) / jnp.sqrt(D)
    attn = jax.nn.softmax(logits, axis=-1)
    synced = jnp.matmul(attn, V)
    synced = jnp.dot(synced, sync_params["W_o"])

    # Reshape back: (B, L, num_branches, D)
    synced_thoughts = synced.reshape(B, L, num_branches, D)

    # Residual connection from pre-sync
    final_thoughts = final_thoughts + synced_thoughts * 0.5  # Partial integration

    # 4. ORTHOGONALITY LOSS (CRITICAL FOR CREATIVITY)
    # Branch'lerin birbirinden farklı vektörler üretmesini zorlarız.
    # Cosine similarity matrix'in off-diagonal elemanlarını minimize ederiz.
    # Mean thought vector over L: (B, num_branches, D)
    mean_thoughts = jnp.mean(final_thoughts, axis=1)
    # Normalize vectors
    norms = jnp.linalg.norm(mean_thoughts, axis=-1, keepdims=True) + 1e-6
    normalized = mean_thoughts / norms
    # Cosine Matrix: (B, num_branches, num_branches)
    cosine_mat = jnp.matmul(normalized, normalized.transpose(0, 2, 1))
    # Identity matrix (diagonals are 1, we want 0 elsewhere)
    eye = jnp.eye(num_branches)[None, :, :]
    # Loss = Mean squared error of off-diagonals
    ortho_loss = jnp.mean((cosine_mat - eye) ** 2)

    # 5. CREATIVE MERGER (Convergence)
    # Main input (x) acts as Query to select best ideas from Branches (Keys)
    merger_params = params["merger"]

    query = jnp.dot(x, merger_params["query_proj"]) # (B, L, D)

    # Use synced thoughts as source of ideas
    keys = jnp.dot(final_thoughts, merger_params["key_proj"]) # (B, L, num_branches, D)

    # Attention: "Hangi düşünce dalı şu anki duruma en uygun?"
    # Score: (B, L, 1, D) dot (B, L, num_branches, D)^T -> (B, L, 1, num_branches)
    scores = jnp.einsum('bld,blnd->bln', query, keys) / jnp.sqrt(D)
    weights = jax.nn.softmax(scores, axis=-1) # (B, L, N)

    # Weighted sum of thoughts
    merged = jnp.einsum('bln,blnd->bld', weights, final_thoughts)

    # Final projection
    out = layer_norm(merged, params["output_norm"])
    out = jnp.dot(out, params["output_proj"])

    return x + out, ortho_loss


# Enhanced reasoning class for integration
class HyperCognitive:
    """HyperCognitive reasoning engine with orthogonality loss."""

    def __init__(self, d_model: int = 512, num_branches: int = 4, max_steps: int = 8, key: random.PRNGKey = random.PRNGKey(42)):
        self.params = hypercognitive_init(d_model, num_branches, max_steps, key)

    def forward(self, x: jnp.ndarray, effort: float = 0.7, train: bool = True) -> Tuple[jnp.ndarray, float]:
        return hypercognitive_apply(self.params, x, effort, train)
