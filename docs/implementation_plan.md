# AGIFORMER v2.0 Implementation Plan

**Goal:** Upgrade AGIFORMER architecture to address "Blind Memory", "Patch Discontinuity", and "Blind Reasoning" flaws.

## User Review Required
> [!IMPORTANT]
> **Breaking Changes:**
> *   **Decoder:** Switching from GRU to **Parallel MLP Decoder** for speed and stability.
> *   **Memory:** `decay_logits` is now input-dependent (dynamic).
> *   **Encoder:** Patching now overlaps (`kernel_size=6`, `stride=4`).
> *   **Normalization:** Switching from `LayerNorm` to `RMSNorm`.
> *   **Activation:** Switching from `GELU` to `SwiGLU`.

## Proposed Changes

### `src/models/encoder.py`
*   [MODIFY] `ByteLatentEncoder`:
    *   Change `Conv1d` kernel size to 6 (Overlap).
    *   Add `RMSNorm` class (or use `nn.RMSNorm` if available, else implement).

### `src/models/memory.py`
*   [MODIFY] `HebbianMemory`:
    *   Remove static `decay_logits`.
    *   Add `decay_net` (Linear layer) to predict decay from input $x_t$.
    *   Update forward pass to use dynamic decay.

### `src/models/agiformer.py`
*   [MODIFY] `LocalAutoregressiveHead`:
    *   Replace `nn.GRU` with `nn.Linear` (MLP Decoder).
    *   Predict 4 bytes in parallel from the latent vector.
*   [MODIFY] `AGIFORMER`:
    *   Update initialization to reflect new components.

### `src/models/reasoning.py`
*   [MODIFY] `RecurrentReasoningBlock`:
    *   Add `CrossAttention` layer.
    *   Update forward pass to attend to `memory_context` (if available) or just self-refinement. *Note: For simplicity and speed, we might stick to self-refinement with Adaptive Computation Time (ACT) first, or just add a simple cross-attention if the memory state is exposed.*
    *   **Decision:** Let's implement **Adaptive Computation Time (ACT)** logic (exit gate) as requested.

### `src/models/layers.py`
*   [MODIFY] `HybridBlock`:
    *   Implement **Gated Fusion**: `gate = sigmoid(Linear(x))`.
    *   Replace `LayerNorm` with `RMSNorm`.
    *   Replace `GELU` MLP with `SwiGLU`.

## Verification Plan

### Automated Tests
1.  **Shape Check:** Run a dummy forward pass with `d_model=512` to ensure no dimension mismatches.
2.  **Overfit Test:** Run `overfit_test.py` on a small batch to verify convergence capability.

### Manual Verification
*   Review code changes.
*   User will restart training using `train_scaled.py` (updated for v2).
