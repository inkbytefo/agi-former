# Training Guide v2.0

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 16GB+ GPU memory (T4/A10G) recommended for Scaled Model
- ~500MB disk space (Turkish Wikipedia dataset)

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/inkbytefo/agi-former.git
cd agi-former
pip install -r requirements.txt
```

### 2. Run Training (Scaled v2.0)
```bash
# Runs in background, logs to file
nohup python -u train_scaled.py > training_v2.log 2>&1 &
tail -f training_v2.log
```

**Expected Output:**
```
Step 10: Loss = 5.8451 | BPC = 8.1056 | LR = 3.00e-05
...
Step 1000: Loss = 1.9234 | BPC = 2.7745 | LR = 2.00e-04
...
Step 50000: Loss = 1.0188 | BPC = 1.4500 | LR = 1.00e-05
```

**Training Time:** ~5-7 days (T4 GPU, 50K steps)

---

## Configuration

Edit hyperparameters in `train_scaled.py`:

```python
# Scaled Model (129M Params)
D_MODEL = 768
N_LAYERS = 12
NUM_HEADS = 12
PATCH_SIZE = 4
WINDOW_SIZE = 256
THINKING_STEPS = 3

# Training
BATCH_SIZE = 2          # Physical batch size
ACCUM_STEPS = 4         # Effective batch size = 8
MAX_STEPS = 50000
BASE_LR = 2e-4
WARMUP_STEPS = 500
GRAD_CLIP = 0.5
```

### Hyperparameter Guide

#### Model Size
- **v1.0 (Small):** `d_model=512, n_layers=6` → Proof of Concept
- **v2.0 (Scaled):** `d_model=768, n_layers=12` → **Current Standard**

#### System 2
- `thinking_steps=3` → **Default** (Active Reasoning with ACT)

---

## Dataset

### Turkish Wikipedia (trwiki)
- Automatically downloaded and cleaned via `src/data/clean_turkish_data.py`.
- Streaming implementation (no massive download needed).

---

## Training Process

### 1. Initialization
```
[*] Creating AGIFORMER v2.0 model...
    - d_model=768, n_layers=12, thinking_steps=3
[*] Parameters: ~129M
```

### 2. Warmup Phase
- Linear LR ramp: `0 → 2e-4`
- High loss expected initially.

### 3. Learning Phase
- Loss decreases steadily.
- Validation every 2000 steps.
- Checkpoints saved as `checkpoint_step_N.pth`.

### 4. Checkpointing
- `best_model_scaled.pth` → Lowest validation loss.
- `last_model_scaled.pth` → Final checkpoint.

---

## Monitoring

### Metrics

**Loss:** Cross-entropy (lower is better)
**BPC (Bits Per Character):** `Loss / ln(2)`

### Expected Progress (v2.0)

| Steps | BPC | Status |
|-------|-----|--------|
| 0-1K | 8.0-4.0 | Warmup |
| 5K | 3.0-2.5 | Syntax Learning |
| 20K | 2.0-1.8 | Semantic Emergence |
| 50K | <1.5 | High Fluency |

---

## Troubleshooting

### NaN Loss
**Status:** Solved in v1.0 via `float32` memory bypass.
If it reoccurs:
- Check `HebbianMemory` implementation.
- Reduce Learning Rate.

### Out of Memory
**Solutions:**
- Reduce `BATCH_SIZE` (2 → 1).
- Increase `ACCUM_STEPS` to compensate.
- Enable Gradient Checkpointing (if implemented).

---

## Resuming Training

To continue from checkpoint:

```python
# In train_scaled.py, logic is already present to load 'last_model_scaled.pth' if exists
# (Ensure you implement this logic if not already there)
```

---

## Export to Hugging Face

```bash
python upload_to_hf.py --repo YOUR_USERNAME/agiformer --token YOUR_HF_TOKEN
```
