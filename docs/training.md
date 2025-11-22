# Training Guide

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 6GB+ GPU memory (for batch_size=4)
- ~200MB disk space (enwik8 dataset)

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/inkbytefo/agi-former.git
cd agi-former
pip install -r requirements.txt
```

### 2. Run Training
```bash
python train.py
```

**Expected Output:**
```
Step 10: Loss = 2.8451 | BPC = 4.1056 | LR = 3.00e-05
Step 20: Loss = 2.5123 | BPC = 3.6246 | LR = 6.00e-05
...
Step 5000: Loss = 1.3988 | BPC = 2.0181 | LR = 3.00e-04
-- VALIDATION: Loss = 1.5650 | BPC = 2.2578 --
Saved best_model.pth
```

**Training Time:** ~15 minutes (T4 GPU, 5000 steps)

---

## Configuration

Edit hyperparameters in `train.py`:

```python
# Model
D_MODEL = 512
N_LAYERS = 6
NUM_HEADS = 8
PATCH_SIZE = 4
WINDOW_SIZE = 128
THINKING_STEPS = 3  # System 2 iterations

# Training
BATCH_SIZE = 4
MAX_STEPS = 5000
BASE_LR = 3e-4
WARMUP_STEPS = 100
GRAD_CLIP = 0.5
```

### Hyperparameter Guide

#### Model Size
- **Small:** `d_model=256, n_layers=4` → Fast, lower quality
- **Medium:** `d_model=512, n_layers=6` → **Default** (balanced)
- **Large:** `d_model=768, n_layers=8` → Better BPC, slower

#### System 2
- `thinking_steps=0` → Disable (baseline)
- `thinking_steps=3` → **Default** (active reasoning)
- `thinking_steps=5+` → More refinement, higher compute

---

## Dataset

### Enwik8
**Source:** First 100MB of English Wikipedia XML  
**Size:** 100,000,000 bytes  
**Split:**
- Train: 90MB
- Validation: 5MB
- Test: 5MB

**Auto-download:** Dataset downloads automatically on first run to `./data/enwik8`.

### Custom Data

To train on your own data:

```python
# In train.py, replace:
from src.data.real_data import get_enwik8_dataloader

# With your custom loader:
def get_custom_dataloader(batch_size, seq_len):
    # Your implementation
    # Must return: (batch, seq_len) tensors of bytes (0-255)
    pass
```

---

## Training Process

### 1. Initialization
```
[*] Creating AGIFORMER model...
    - d_model=512, n_layers=6, thinking_steps=3
[*] Parameters: ~50M
[*] Downloading enwik8... (if first run)
```

### 2. Warmup Phase (Steps 0-100)
```
Step 10: Loss = 2.8451 | BPC = 4.1056 | LR = 3.00e-05
```
- Linear LR ramp: `0 → 3e-4`
- High loss expected (model random)

### 3. Learning Phase (Steps 100-5000)
```
Step 1000: Loss = 1.9234 | BPC = 2.7745 | LR = 3.00e-04
Step 2000: Loss = 1.7123 | BPC = 2.4701 | LR = 3.00e-04
Step 3000: Loss = 1.6234 | BPC = 2.3418 | LR = 3.00e-04
```
- Loss decreases steadily
- Validation every 200 steps

### 4. Checkpointing
```
-- VALIDATION: Loss = 1.5650 | BPC = 2.2578 --
Saved best_model.pth
```
- `best_model.pth` → Lowest validation loss
- `last_model.pth` → Final checkpoint

---

## Monitoring

### Metrics

**Loss:** Cross-entropy (lower is better)
```
Loss = -log P(next_byte | context)
```

**BPC (Bits Per Character):** 
```
BPC = Loss / ln(2)
```
- Random baseline: 8.0 BPC
- Character-level models: 1.2-1.5 BPC
- AGIFORMER (5k steps): 2.26 BPC

### Expected Progress

| Steps | BPC | Status |
|-------|-----|--------|
| 0-100 | 4.0-3.5 | Warmup |
| 500 | 3.0-2.8 | Learning syntax |
| 1000 | 2.8-2.6 | Basic patterns |
| 3000 | 2.5-2.3 | Word structure |
| 5000 | 2.3-2.2 | ✅ Proof of concept |
| 20k+ | <2.0 | Production quality |

---

## Troubleshooting

### NaN Loss
**Symptoms:**
```
Step 150: Loss = nan | BPC = nan
```

**Causes:**
1. Learning rate too high
2. Gradient explosion
3. Numerical instability in attention

**Solutions:**
- ✅ Already fixed in code (stability patches)
- If persists: Lower `BASE_LR` to `1e-4`
- Increase `GRAD_CLIP` to `1.0`

### Out of Memory
**Error:**
```
CUDA out of memory
```

**Solutions:**
- Reduce `BATCH_SIZE` (4 → 2 → 1)
- Reduce `d_model` (512 → 256)
- Reduce `n_layers` (6 → 4)

### Slow Training
**<100 steps/min:**

**Solutions:**
- Use GPU (not CPU): `DEVICE = 'cuda'`
- Enable mixed precision: `torch.cuda.amp.autocast()`
- Reduce `thinking_steps` (3 → 1)

---

## Advanced: Multi-GPU

For distributed training:

```python
# In train.py
import torch.distributed as dist

# Wrap model
model = torch.nn.parallel.DistributedDataParallel(model)

# Launch
torchrun --nproc_per_node=4 train.py
```

**Expected Speedup:** ~3.5× on 4 GPUs

---

## Resuming Training

To continue from checkpoint:

```python
# In train.py, after model creation:
if os.path.exists("last_model.pth"):
    model.load_state_dict(torch.load("last_model.pth"))
    print("Resumed from checkpoint")
```

---

## Hyperparameter Tuning

### Learning Rate
- **Too High (>5e-4):** Loss spikes, NaN
- **Too Low (<1e-5):** Slow convergence
- **Sweet Spot:** `3e-4` with warmup

### Gradient Clipping
- **Too Aggressive (<0.1):** Slow learning
- **Too Loose (>2.0):** Instability
- **Default:** `0.5`

### System 2 Steps
- `0`: Baseline (no thinking)
- `1-3`: **Recommended** (active reasoning)
- `5+`: Diminishing returns (expensive)

---

## Export to Hugging Face

```bash
python upload_to_hf.py --repo YOUR_USERNAME/agiformer --token YOUR_HF_TOKEN
```

Uploads:
- `best_model.pth`
- Source code (`src/`)
- Documentation

---

## Next Steps

After training:
1. **Test Generation:** `python generate.py`
2. **Inspect System 2:** `python inspect_reasoning.py`
3. **Extend Training:** Increase `MAX_STEPS` to 20k+
4. **Fine-tune:** Change dataset to your domain
