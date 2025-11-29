## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp
from jax import random
from .encoder import encoder_init, encoder_apply
from .layers import hybrid_block_init, hybrid_block_apply
from .reasoning import reasoning_init, reasoning_apply
from .hypercognitive import hypercognitive_init, hypercognitive_apply
from .strange_attractor import strange_attractor_init, strange_attractor_apply

def byte_head_init(d_model, patch_size, key: random.PRNGKey):
    k1, k2, k3 = random.split(key, 3)
    W1 = random.normal(k1, (d_model, 512)) * (1.0 / jnp.sqrt(d_model))
    b1 = jnp.zeros((512,))
    W2 = random.normal(k2, (512, 512)) * (1.0 / jnp.sqrt(512))
    b2 = jnp.zeros((512,))
    W3 = random.normal(k3, (512, patch_size * 256)) * (1.0 / jnp.sqrt(512))
    b3 = jnp.zeros((patch_size * 256,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "patch_size": patch_size}

def byte_head_apply(params, latents):
    B, N, D = latents.shape
    h1 = jax.nn.gelu(jnp.einsum('bnd,df->bnf', latents, params["W1"]) + params["b1"]) 
    h2 = jax.nn.gelu(jnp.einsum('bnf,fg->bng', h1, params["W2"]) + params["b2"]) 
    logits = jnp.einsum('bng,gh->bnh', h2, params["W3"]) + params["b3"] 
    logits = logits.reshape(B, N, params["patch_size"], 256) 
    return logits

def morph_head_init(d_model, root_vocab, suffix_vocab, suffix_slots, key: random.PRNGKey):
    k_root1, k_root2, k_slots, k_slot_emb = random.split(key, 4)
    W1_root = random.normal(k_root1, (d_model, 512)) * (1.0 / jnp.sqrt(d_model))
    b1_root = jnp.zeros((512,))
    W2_root = random.normal(k_root2, (512, root_vocab)) * (1.0 / jnp.sqrt(512))
    b2_root = jnp.zeros((root_vocab,))
    slot_W1 = random.normal(k_slots, (suffix_slots, d_model, 512)) * (1.0 / jnp.sqrt(d_model))
    slot_b1 = jnp.zeros((suffix_slots, 512))
    slot_W2 = random.normal(random.fold_in(key, 99), (suffix_slots, 512, suffix_vocab)) * (1.0 / jnp.sqrt(512))
    slot_b2 = jnp.zeros((suffix_slots, suffix_vocab))
    slot_emb = random.normal(k_slot_emb, (suffix_slots, d_model)) * (1.0 / jnp.sqrt(d_model))
    return {
        "W1_root": W1_root, "b1_root": b1_root, "W2_root": W2_root, "b2_root": b2_root,
        "slot_W1": slot_W1, "slot_b1": slot_b1, "slot_W2": slot_W2, "slot_b2": slot_b2,
        "slot_emb": slot_emb, "suffix_slots": suffix_slots,
    }

def morph_head_apply(params, latents):
    B, N, D = latents.shape
    h1 = jax.nn.gelu(jnp.einsum('bnd,df->bnf', latents, params["W1_root"]) + params["b1_root"]) 
    root_logits = jnp.einsum('bnf,fg->bng', h1, params["W2_root"]) + params["b2_root"]
    S = params["suffix_slots"]
    slot_emb = params["slot_emb"]  # (S, D)
    # Add slot embedding and apply per-slot classifiers
    def one_slot(s):
        x_s = latents + slot_emb[s][None, None, :]
        h_s = jax.nn.gelu(jnp.einsum('bnd,df->bnf', x_s, params["slot_W1"][s]) + params["slot_b1"][s])
        log_s = jnp.einsum('bnf,fg->bng', h_s, params["slot_W2"][s]) + params["slot_b2"][s]
        return log_s
    suffix_logits = jnp.stack([one_slot(s) for s in range(S)], axis=2)  # (B, N, S, Vsuffix)
    return {"root": root_logits, "suffix": suffix_logits}

def agiformer_init(d_model=512, n_layers=6, num_heads=8, patch_size=4, window_size=128, thinking_steps=3, root_vocab_size=50000, suffix_vocab_size=1000, suffix_slots=5, enable_hypercognitive=False, num_branches=4, enable_strange_attractor=False, attractor_steps=8, key: random.PRNGKey = random.PRNGKey(0)):
    enc = encoder_init(d_model, patch_size, random.fold_in(key, 1))
    # Override vocab sizes in encoder if needed, or just use the passed args for heads
    enc["root_vocab_size"] = root_vocab_size
    enc["suffix_vocab_size"] = suffix_vocab_size

    layers = [hybrid_block_init(d_model, num_heads, window_size, random.fold_in(key, 100 + i)) for i in range(n_layers)]
    norm_gamma = jnp.ones((d_model,))
    norm_beta = jnp.zeros((d_model,))
    reason = reasoning_init(d_model, random.fold_in(key, 999))

    # HyperCognitive integration
    hypercognitive = None
    if enable_hypercognitive:
        hypercognitive = hypercognitive_init(d_model, num_branches, thinking_steps * 2, random.fold_in(key, 2000))

    byte_head = byte_head_init(d_model, patch_size, random.fold_in(key, 1000))

    morph_head = morph_head_init(d_model, root_vocab_size, suffix_vocab_size, suffix_slots, random.fold_in(key, 1001))
    return {
        "encoder": enc, "layers": layers, "norm_gamma": norm_gamma, "norm_beta": norm_beta,
        "reason": reason, "hypercognitive": hypercognitive, "byte_head": byte_head, "morph_head": morph_head,
        "thinking_steps": thinking_steps, "suffix_slots": suffix_slots, "enable_hypercognitive": enable_hypercognitive, "num_branches": num_branches
    }

def layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    xhat = (x - mean) / jnp.sqrt(var + 1e-5)
    return gamma * xhat + beta

def agiformer_apply(params, x, effort: float = 1.0, train: bool = True) -> Tuple[Any, float]:
    raw_ndim = x.ndim
    x = encoder_apply(params["encoder"], x)
    for layer in params["layers"]:
        x = hybrid_block_apply(layer, x, effort)
    x = layer_norm(x, params["norm_gamma"], params["norm_beta"])

    loss = 0.0  # Default no loss
    # Reasoning: Use HyperCognitive if enabled, else basic reasoning
    if params["enable_hypercognitive"] and params["hypercognitive"] is not None:
        x, ortho_loss = hypercognitive_apply(params["hypercognitive"], x, effort, train)
        loss += ortho_loss
    else:
        x = reasoning_apply(params["reason"], x, params["thinking_steps"], effort)

    # Strange Attractor: Bilinç döngüsü ve kaos kenarı dengesi
    if params.get("enable_strange_attractor", False) and params["strange_attractor"] is not None:
        x, stability_loss = strange_attractor_apply(params["strange_attractor"], x, train)
        loss += 0.05 * stability_loss  # lambda_stability = 0.05

    if raw_ndim == 2:
        output = byte_head_apply(params["byte_head"], x)
    else:
        output = morph_head_apply(params["morph_head"], x)

    return output, loss

class AGIFORMER:
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        num_heads: int = 8,
        patch_size: int = 4,
        window_size: int = 128,
        thinking_steps: int = 3,
        enable_hypercognitive: bool = False,
        num_branches: int = 4,
        enable_strange_attractor: bool = False,
        attractor_steps: int = 8,
        key: random.PRNGKey = random.PRNGKey(0)
    ):
        self.params = agiformer_init(d_model, n_layers, num_heads, patch_size, window_size, thinking_steps,
                                   enable_hypercognitive=enable_hypercognitive, num_branches=num_branches,
                                   enable_strange_attractor=enable_strange_attractor, attractor_steps=attractor_steps, key=key)

    def forward(self, x, effort: float = 1.0, train: bool = True):
        output, ortho_loss = agiformer_apply(self.params, x, effort, train)
        # For backward compatibility, return just output unless train flag indicates otherwise
        if train and self.params["enable_hypercognitive"]:
            return output, ortho_loss
        return output

    def to(self, device):
        return self

    def eval(self):
        return None

    def load_state_dict(self, state):
        return None
