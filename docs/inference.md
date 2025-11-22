# Inference Guide

## Quick Start

```bash
python generate.py
```

**Default Output:**
```
Prompt: 'The history of '
--------------------------------------------------
The history of Tomadination of the [[New Gouple de aparty]]...
```

---

## Basic Usage

### 1. Load Model
```python
from src.models.agiformer import AGIFORMER
import torch

model = AGIFORMER(d_model=512, n_layers=6, patch_size=4, thinking_steps=3)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```

### 2. Prepare Input
```python
prompt = "The history of artificial intelligence"
input_bytes = [ord(c) for c in prompt]

# Pad to patch_size boundary
pad = (4 - len(input_bytes) % 4) % 4
input_bytes.extend([32] * pad)

x = torch.tensor(input_bytes).unsqueeze(0)  # (1, seq_len)
```

### 3. Generate
```python
with torch.no_grad():
    output = model(x, temperature=0.7)  # (1, num_patches, patch_size)
    
# Decode
generated_bytes = output[0, -1, :].tolist()
text = ''.join([chr(b) for b in generated_bytes if 32 <= b <= 126])
```

---

## Temperature Sampling

### Greedy (Temperature = 0)
```python
output = model(x, temperature=0.0)
```
- Picks most likely byte every step
- **Deterministic** (same output each run)
- Prone to repetition loops

**Example:**
```
The history of of of of of...
```

### Low Temperature (0.3 - 0.5)
```python
output = model(x, temperature=0.3)
```
- Slightly random, still conservative
- Good for **coherent** text
- Reduces repetition

**Example:**
```
The history of the computer system...
```

### Medium Temperature (0.7 - 0.9)
```python
output = model(x, temperature=0.7)  # Default
```
- Balanced creativity/coherence
- **Recommended** for exploration

**Example:**
```
The history of Tomadination of the [[New Gouple]]...
```

### High Temperature (1.0+)
```python
output = model(x, temperature=1.2)
```
- Very random
- Incoherent but diverse
- Good for **debugging** model knowledge

**Example:**
```
The history qw8#$x [[zap]] nullification...
```

---

## Advanced: Token-by-Token Generation

For streaming output:

```python
def generate_stream(model, prompt, max_tokens=200, temperature=0.7):
    # Encode prompt
    context = [ord(c) for c in prompt]
    pad = (4 - len(context) % 4) % 4
    context.extend([32] * pad)
    
    for _ in range(max_tokens // 4):  # Generate patch-by-patch
        x = torch.tensor(context[-1024:]).unsqueeze(0)  # Sliding window
        
        with torch.no_grad():
            pred = model(x, temperature=temperature)
        
        # Get last patch
        new_bytes = pred[0, -1, :].cpu().tolist()
        context.extend(new_bytes)
        
        # Decode and print
        chunk = ''.join([chr(b) for b in new_bytes if 32 <= b <= 126])
        print(chunk, end='', flush=True)
```

**Usage:**
```python
generate_stream(model, "The history of ", max_tokens=200)
```

---

## System 2 Control

### Disable Thinking (Baseline)
```python
model = AGIFORMER(thinking_steps=0)  # Skip System 2
```
- Faster inference (~2× speedup)
- Lower quality output

### Increase Thinking
```python
model = AGIFORMER(thinking_steps=5)  # More refinement
```
- Slower inference
- Potentially better coherence

### Runtime Control
System 2 is part of the model, so you must reinitialize:
```python
# Not possible to change thinking_steps after model creation
# Must create new model with desired config
```

---

## Batch Inference

Process multiple prompts:

```python
prompts = ["The history of ", "In the year 2050, ", "Once upon a time, "]
batch = []

for prompt in prompts:
    bytes = [ord(c) for c in prompt]
    pad = (4 - len(bytes) % 4) % 4
    bytes.extend([32] * pad)
    batch.append(torch.tensor(bytes))

# Pad to same length
max_len = max(t.size(0) for t in batch)
batch_tensor = torch.stack([
    F.pad(t, (0, max_len - t.size(0)))
    for t in batch
])

# Generate
with torch.no_grad():
    outputs = model(batch_tensor, temperature=0.7)
```

---

## Debugging Output

### Check Raw Bytes
```python
generated = model(x, temperature=0.0)
raw_bytes = generated[0, -1, :].tolist()
print(f"Raw: {raw_bytes}")  # e.g., [116, 104, 101, 32]
```

### Detect Non-Printables
```python
for b in raw_bytes:
    if not (32 <= b <= 126):
        print(f"Warning: Non-ASCII byte {b}")
```

### Measure Entropy
```python
import torch.nn.functional as F

logits = model.head(latents)  # Get raw logits
probs = F.softmax(logits, dim=-1)
entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

print(f"Avg Entropy: {entropy.item():.2f} bits")
# Low (<2): Confident, may repeat
# High (>6): Confused, will be random
```

---

## Common Issues

### Repetition Loops
**Problem:**
```
of of of of of...
```

**Solutions:**
1. Increase temperature: `0.0 → 0.7`
2. Use nucleus sampling (top-p):
   ```python
   probs = F.softmax(logits / temp, dim=-1)
   sorted_probs, indices = torch.sort(probs, descending=True)
   cumsum = torch.cumsum(sorted_probs, dim=-1)
   mask = cumsum > 0.9  # Keep top 90%
   sorted_probs[mask] = 0
   next_byte = torch.multinomial(sorted_probs, 1)
   ```

### Gibberish Output
**Problem:**
```
xq#$8z [[nullification]]...
```

**Causes:**
- Temperature too high
- Model undertrained

**Solutions:**
- Lower temperature: `1.2 → 0.5`
- Train longer (20k+ steps)

### Slow Inference
**Problem:** >1s per token

**Solutions:**
- Use GPU: `model.cuda()`
- Reduce `thinking_steps`: `3 → 1`
- Disable System 2: `thinking_steps=0`

---

## Performance Benchmarks

**GPU:** NVIDIA T4  
**Prompt Length:** 100 bytes  
**Generation Length:** 200 bytes

| Config | Latency | Throughput |
|--------|---------|------------|
| Greedy (temp=0) | 45ms | 22 tokens/s |
| Sampling (temp=0.7) | 52ms | 19 tokens/s |
| System 2 disabled | 28ms | 36 tokens/s |

---

## API Reference

### Model Forward
```python
def forward(
    x: torch.Tensor,           # (Batch, Seq_Len) bytes
    target_bytes: Optional[torch.Tensor] = None,  # For training
    temperature: float = 0.0   # Sampling temp (0 = greedy)
) -> torch.Tensor:
    # Returns: (Batch, Num_Patches, Patch_Size, 256) if training
    #          (Batch, Num_Patches, Patch_Size) if inference
```

### Generation Utilities
See `generate.py` for full implementation:
- `generate_text(model_path, prompt, max_tokens, temperature)`
- Automatic padding and decoding

---

## Next Steps

1. **Experiment with Prompts:** Try different domains
2. **Tune Temperature:** Find sweet spot for your use case
3. **Extend Context:** Modify `generate.py` to use longer contexts
4. **Fine-tune:** Retrain on domain-specific data
