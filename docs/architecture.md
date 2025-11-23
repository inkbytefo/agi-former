# Architecture Guide v2.0

## Overview

AGIFORMER v2.0 implements a novel hybrid architecture combining byte-level processing, linear attention, and iterative reasoning with adaptive computation.

## Pipeline Flow

```
Input Bytes
    ↓
ByteLatentEncoder (Soft Patching + RoPE)
    ↓
HybridBlock × N (Gated Fusion: Linear Attn + Sliding Window + Hebbian Memory)
    ↓
RecurrentReasoningBlock (System 2 - ACT with Exit Gate)
    ↓
LocalAutoregressiveHead (Parallel MLP Decoder)
    ↓
Output Bytes
```

---

## 1. ByteLatentEncoder

**File:** `src/models/encoder.py`

### Purpose
Converts raw byte sequences into latent patches with positional information, using soft boundaries to prevent discontinuities.

### Architecture (v2.0)
- **Input:** `(Batch, Seq_Len)` bytes (0-255)
- **Embedding:** `nn.Embedding(256, d_model)`
- **Soft Patching:** `Conv1d(kernel=6, stride=4)` -> Overlap of 2 bytes.
- **RoPE:** Rotary Positional Embeddings for length generalization.
- **Normalization:** RMSNorm.
- **Output:** `(Batch, Num_Patches, d_model)`

### Key Design Decisions
- **Why Soft Patching?** Prevents "stuttering" at patch boundaries by allowing information to bleed across patches.
- **Why RoPE?** Enables extrapolation to longer sequences than training.

---

## 2. HybridBlock

**File:** `src/models/layers.py`

### Components

#### 2.1 LinearAttention
**Complexity:** $O(N)$ instead of $O(N^2)$

**Formula:**
```
Q = elu(Wq * x) + 1.0 + ε
K = elu(Wk * x) + 1.0 + ε
V = Wv * x

Attention(Q, K, V) = (Q @ cumsum(K ⊗ V)) / (Q @ cumsum(K) + ε)
```

#### 2.2 SlidingWindowAttention
**Complexity:** $O(N × window_size)$
**Purpose:** High-precision local context.

#### 2.3 HebbianMemory (Input-Dependent)
**File:** `src/models/memory.py`

**v2.0 Upgrade:**
- **Old:** Static decay $\lambda$.
- **New:** Input-Dependent Decay $\lambda_t = \sigma(W_{decay} x_t)$.
- **Effect:** Model can selectively "lock" important information (e.g., names, dates) while forgetting noise.

### Fusion (Gated)
```python
g = sigmoid(Gate(x))
output = g * LocalAttn + (1-g) * GlobalMemory
```
- Allows the model to dynamically choose between local precision and global context.

### MLP (SwiGLU)
- Replaced GELU with SwiGLU for better capacity.
- Replaced LayerNorm with RMSNorm for stability.

---

## 3. RecurrentReasoningBlock (System 2)

**File:** `src/models/reasoning.py`

### Purpose
"Thinking for time" - Iterative refinement of latent representations.

### Mechanism (ACT - Adaptive Computation Time)
```python
for step in range(max_steps=3):
    # Refine
    z_new = z + MLP(RMSNorm(z))
    
    # Exit Gate
    p_halt = sigmoid(HaltNet(z))
    
    # Soft Update
    z = z + (1 - p_halt) * update
```
- **Effect:** If `p_halt` is high (confident), the state stops updating.
- **Benefit:** Efficient compute allocation.

---

## 4. LocalAutoregressiveHead (Decoder)

**File:** `src/models/agiformer.py`

### Purpose
Decodes latent patches into byte sequences.

### Architecture (v2.0 - Parallel MLP)
- **Old:** GRU (Sequential, Slow).
- **New:** Parallel MLP.
- **Input:** Latent Vector $z$ (d_model).
- **Output:** $4 \times 256$ logits (predicts 4 bytes at once).

**Why MLP?**
- Removes the sequential bottleneck of GRU.
- Allows the global model to fully control the patch content.
- Faster training and inference.

---

## Loss Function

**Training:** Cross-entropy on next-patch prediction
```python
loss = CrossEntropy(logits, targets)
BPC = loss / ln(2)  # Bits per character
```

**Metric:** BPC (Bits Per Character) - lower is better
- Random baseline: 8.0 BPC
- Good model: < 1.5 BPC
- AGIFORMER v1.0: 1.85 BPC
- AGIFORMER v2.0 Target: < 1.5 BPC

---

## Hyperparameters (Scaled v2.0)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `d_model` | 768 | Increased capacity (100M+) |
| `n_layers` | 12 | Deeper reasoning |
| `num_heads` | 12 | Aspect ratio maintained |
| `patch_size` | 4 | 4× compression |
| `window_size` | 256 | Wider local context |
| `thinking_steps` | 3 | System 2 iterations (ACT) |
| `learning_rate` | 2e-4 | Conservative for scale |
| `batch_size` | 2 | T4 memory limit (Accum=4) |

---

## References

- **Linear Transformers:** Katharopoulos et al., 2020
- **RoPE:** Su et al., 2021
- **System 2 Deep Learning:** Bengio et al., 2019
- **Mamba:** Gu & Dao, 2023 (Input-Dependent Decay inspiration)
- **SwiGLU:** Shazeer, 2020

