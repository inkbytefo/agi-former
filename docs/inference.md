# Inference Guide v2.0

## Quick Start

```bash
python generate.py best_model_scaled.pth
```

**Default Output:**
```
Prompt: 'The history of '
--------------------------------------------------
The history of the Ottoman Empire began with the...
```

---

## Basic Usage

### 1. Load Model (v2.0 Scaled)
```python
from src.models.agiformer import AGIFORMER
import torch

# v2.0 Configuration
model = AGIFORMER(
    d_model=768, 
    n_layers=12, 
    patch_size=4, 
    thinking_steps=3
).to('cuda')

model.load_state_dict(torch.load("best_model_scaled.pth"))
model.eval()
```

### 2. Prepare Input
```python
prompt = "The history of artificial intelligence"
input_bytes = [ord(c) for c in prompt]

# Pad to patch_size boundary
pad = (4 - len(input_bytes) % 4) % 4
input_bytes.extend([32] * pad)

x = torch.tensor(input_bytes).unsqueeze(0).to('cuda')  # (1, seq_len)
```

### 3. Generate
```python
with torch.no_grad():
    # v2.0 returns logits directly
    logits = model(x)  # (1, num_patches, 4, 256)
    
# Decode
last_patch_logits = logits[0, -1, :, :] # (4, 256)
generated_bytes = torch.argmax(last_patch_logits, dim=-1).tolist()
text = ''.join([chr(b) for b in generated_bytes if 32 <= b <= 126])
```

---

## Temperature Sampling

### Greedy (Temperature = 0)
- Picks most likely byte every step.
- **Deterministic**.

### Sampling (Temperature > 0)
- Requires manual sampling logic on logits since v2.0 `forward` returns raw logits in inference mode too (unlike v1.0 which had built-in sampling).

```python
def sample(logits, temperature=0.7):
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
```

---

## Advanced: Token-by-Token Generation

For streaming output with v2.0 (Parallel Decoder):

```python
def generate_stream(model, prompt, max_tokens=200, temperature=0.7):
    context = [ord(c) for c in prompt]
    
    for _ in range(max_tokens // 4):
        # Prepare batch
        x = torch.tensor(context[-1024:]).unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            logits = model(x)
        
        # Get last patch (4 bytes)
        last_patch = logits[0, -1, :, :]
        new_bytes = sample(last_patch, temperature).tolist()
        
        context.extend(new_bytes)
        chunk = ''.join([chr(b) for b in new_bytes if 32 <= b <= 126])
        print(chunk, end='', flush=True)
```

---

## System 2 Control (ACT)

In v2.0, System 2 uses **Adaptive Computation Time (ACT)**.
- The model *automatically* decides how many steps to think (up to `thinking_steps`).
- You can monitor the `halt_probability` if you modify `reasoning.py` to return it.

---

## Troubleshooting

### Repetition Loops
- **Cause:** Greedy sampling on ambiguous prompts.
- **Solution:** Increase temperature to 0.7.

### Gibberish Output
- **Cause:** Model undertrained or temperature too high.
- **Solution:** Train for 50K steps or reduce temperature.

---

## Performance Benchmarks (v2.0)

**GPU:** NVIDIA T4  
**Prompt Length:** 100 bytes  
**Generation Length:** 200 bytes

| Config | Latency | Throughput |
|--------|---------|------------|
| Greedy (temp=0) | 35ms | 28 tokens/s |
| **Speedup vs v1.0** | **+27%** | **Parallel Decoder** |

---

## API Reference

### Model Forward
```python
def forward(
    self, 
    x: torch.Tensor,           # (Batch, Seq_Len)
    target_bytes: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Returns: (Batch, Num_Patches, 4, 256) Logits
```
