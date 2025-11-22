# Architecture Guide

## Overview

AGIFORMER implements a novel hybrid architecture combining byte-level processing, linear attention, and iterative reasoning.

## Pipeline Flow

```
Input Bytes
    ↓
ByteLatentEncoder (with RoPE)
    ↓
HybridBlock × N (Linear Attention + Sliding Window)
    ↓
RecurrentReasoningBlock (System 2 - 3 steps)
    ↓
LocalAutoregressiveHead (GRU-based decoder)
    ↓
Output Bytes
```

---

## 1. ByteLatentEncoder

**File:** `src/models/encoder.py`

### Purpose
Converts raw byte sequences into latent patches with positional information.

### Architecture
- **Input:** `(Batch, Seq_Len)` bytes (0-255)
- **Embedding:** `nn.Embedding(256, d_model)`
- **Patching:** Reshape to `(Batch, Num_Patches, Patch_Size, d_model)`
- **RoPE:** Rotary Positional Embeddings for length generalization
- **Projection:** Linear layer to final latent dimension
- **Output:** `(Batch, Num_Patches, d_model)`

### Key Design Decisions
- **Why RoPE?** Enables extrapolation to longer sequences than training
- **Why Patching?** Reduces sequence length by factor of `patch_size` (default: 4)

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

**Stability Fixes:**
- `elu(x) + 1.0 + 1e-4` ensures strict positivity (prevents division by zero)
- `Q` scaled by `sqrt(head_dim)` to control magnitude
- Layer norm on output

#### 2.2 SlidingWindowAttention
**Complexity:** $O(N × window_size)$

**Implementation:**
```python
scores = (Q @ K.T) / sqrt(d_k)
mask = causal_mask | window_mask  # Blocks far tokens
scores = scores.masked_fill(mask, -1e4)  # Safe masking
attn = softmax(scores)
out = attn @ V
```

**Why Manual?** PyTorch's `scaled_dot_product_attention` was unstable with custom masks.

### Fusion
```python
x = residual + out_proj(attn_out + ssm_out)
```
Parallel branches (not sequential) for efficiency.

---

## 3. RecurrentReasoningBlock (System 2)

**File:** `src/models/reasoning.py`

### Algorithm
```python
z_0 = input  # Initial latent from backbone

for t in range(thinking_steps):
    norm_z = LayerNorm(z_t)
    update = MLP(norm_z)           # Candidate thought
    gate = sigmoid(W_gate @ norm_z) # How much to accept
    z_{t+1} = z_t + gate * update   # Gated residual
    
return z_T  # Refined latent
```

### Design Philosophy
- **Gated Update:** Prevents explosion/vanishing (like LSTM)
- **Residual Connection:** Allows model to skip thinking if not needed
- **Pre-Norm:** Stabilizes deep iteration

### Measured Activity
- **Latent Change:** Δz = 12.7 (Euclidean distance)
- **Gate Bias:** -0.0065 (near neutral)
- **Interpretation:** Model actively refines latents by ~56% per dimension

---

## 4. LocalAutoregressiveHead

**File:** `src/models/agiformer.py`

### Purpose
Decodes latent patches into byte sequences autoregressively.

### Architecture

#### Training Mode
```python
# Teacher forcing
inputs = [SOS, target[0], target[1], ..., target[P-2]]
targets = [target[0], target[1], ..., target[P-1]]

emb = ByteEmb(inputs)                    # (B*N, P, H)
context = LatentProj(latent).expand()     # (B*N, P, H)
rnn_in = concat([emb, context], dim=-1)  # (B*N, P, 2H)

out, _ = GRU(rnn_in)
logits = Linear(out)  # (B*N, P, 256)
```

#### Inference Mode
```python
current = SOS
hidden = None

for i in range(patch_size):
    emb = ByteEmb(current)
    rnn_in = concat([emb, latent_context], dim=-1)
    out, hidden = GRU(rnn_in, hidden)
    logit = Linear(out)
    
    # Sampling
    if temperature > 0:
        next_byte = multinomial(softmax(logit / temp))
    else:
        next_byte = argmax(logit)
    
    current = next_byte
```

### Key Design
- **Concatenation (not Addition):** Preserves signal strength
- **GRU State:** Carries info across steps within a patch
- **Temperature Sampling:** Breaks repetition loops

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
- AGIFORMER: 2.26 BPC (undertrained but stable)

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `d_model` | 512 | Balance capacity/speed |
| `n_layers` | 6 | Deep enough for complexity |
| `num_heads` | 8 | Standard for 512-D |
| `patch_size` | 4 | 4× compression |
| `window_size` | 128 | Local attention context |
| `thinking_steps` | 3 | System 2 iterations |
| `learning_rate` | 3e-4 | With warmup |
| `batch_size` | 4 | GPU memory limit |

---

## Numerical Stability

### Challenges & Solutions

1. **Linear Attention Division by Zero**
   - **Problem:** `elu(x) + 1.0` can = 0 if x very negative
   - **Solution:** `elu(x) + 1.0 + 1e-4` (strict positivity)

2. **SDPA Masking Instability**
   - **Problem:** NaN in `scaled_dot_product_attention` with bool masks
   - **Solution:** Manual attention with `-1e4` instead of `-inf`

3. **System 2 Explosion**
   - **Problem:** Iterative updates could amplify errors
   - **Solution:** Gated residuals + pre-norm + small init

4. **Gradient Clipping**
   - **Value:** 0.5 (aggressive)
   - **Reason:** Prevents spikes during early training

---

## Memory & Compute

**Training (Batch=4, Seq=1024):**
- GPU Memory: ~6 GB (T4)
- Time/Step: ~180ms
- Total for 5000 steps: ~15 min

**Inference (Seq=200):**
- Latency: ~50ms (greedy)
- Memory: ~2 GB

**Scaling:**
- Linear Attention: $O(N)$ time
- System 2: $O(k × N)$ where k = thinking_steps

---

## Comparison to Baselines

| Feature | AGIFORMER | GPT-2 | Mamba |
|---------|-----------|-------|-------|
| Tokenization | None (bytes) | BPE | BPE |
| Attention | Linear ($O(N)$) | Quadratic | N/A |
| Recurrence | System 2 Loop | None | SSM |
| BPC (enwik8) | 2.26 | ~1.1 | ~1.0 |
| Training Time | 15 min | Hours | Hours |

**Note:** BPC gap due to undertrained model, not architecture limit.

---

## Future Improvements

1. **Longer Training:** Target BPC < 1.5
2. **More Thinking Steps:** 3 → 5-7 for harder tasks
3. **Sparse Experts:** Route different "thinking modes"
4. **Memory Module:** External differentiable memory
5. **Multi-Modal:** Extend to images/audio bytes

---

## References

- **Linear Transformers:** Katharopoulos et al., 2020
- **RoPE:** Su et al., 2021
- **System 2 Deep Learning:** Bengio et al., 2019
- **Mamba:** Gu & Dao, 2023
