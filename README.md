# AGIFORMER: Byte-Level Language Model with Neuroplasticity

> **Status:** Phase 9 - Architecture v2.0 Training 🚀 **In Progress**  
> **Latest Achievement:** v2.0 Architecture implemented & verified (Input-Dependent Memory, Soft Patching, ACT)

A research implementation of a byte-level language model featuring:
- 🧠 **Hebbian Memory** with **Input-Dependent Decay** (Selective Forgetting)
- 📚 **Curriculum Learning** (3-stage developmental approach)
- 🔄 **System 2 Reasoning** with **Adaptive Computation Time (ACT)**
- 🚀 **Linear Complexity** attention mechanism
- ⚡ **Parallel MLP Decoder** (No more GRU bottleneck)

## Quick Start

### Installation
```bash
pip install torch datasets tqdm
```

### Training (v2.0 Scaled)
```bash
python train_scaled.py  # 50K steps, 129M params
```

### Inference
```bash
python generate.py best_model_scaled.pth
```

### Testing
```bash
python test_recall_fixed.py  # Memory test (Needle in Haystack)
python overfit_test.py       # Stability verification
```

## Architecture v2.0

```
Bytes → Encoder (Soft Patching) → Hebbian Memory → Reasoning Loop → MLP Decoder → Bytes
         (Overlap=2)              (Input-Dep λ)      (ACT Exit)      (Parallel)
```

### Core Components (v2.0 Upgrades)

- **ByteLatentEncoder:** Soft Patching (Kernel=6, Stride=4) for smoother boundaries.
- **HebbianMemory:** Input-Dependent Decay ($\lambda_t = \sigma(W x_t)$) for selective memory.
- **RecurrentReasoningBlock:** Adaptive Computation Time (ACT) with Exit Gate.
- **LocalAutoregressiveHead:** Parallel MLP Decoder (4x faster than GRU).
- **HybridBlock:** Gated Fusion (Sigmoid) + SwiGLU + RMSNorm.

See [docs/architecture.md](docs/architecture.md) for technical details.

## Features

✅ **No Tokenization** - Universal byte-level processing  
✅ **Linear Complexity** - O(N) attention with Hebbian memory  
✅ **Smart Memory** - Input-Dependent Decay (can "lock" important info)  
✅ **Curriculum Learning** - 3-stage developmental training  
✅ **Adaptive Reasoning** - Dynamic thinking steps (ACT)  
✅ **Modern Components** - SwiGLU, RMSNorm, Soft Patching  

## Curriculum Learning (Phase 7)

### Training Stages

| Stage | Steps | Plasticity (α) | Data | Purpose |
|-------|-------|----------------|------|---------|
| **1. Childhood** | 0-3K | 0.10 | Dictionary | Lexical grounding |
| **2. Youth** | 3K-8K | 0.50 | Stories | Syntactic scaffolding |
| **3. Adulthood** | 8K-20K | 0.99 | Wikipedia | Semantic expansion |

### Results (20K Steps - Turkish Training)

**Metrics:**
- **Final BPC:** 1.85 (↓77% from initialization)
- **Best Val BPC:** 1.78
- **Training Time:** ~50 minutes (CUDA GPU)
- **Stability:** 0 NaN occurrences across 20K steps

**Progress:**
```
Step 0:     BPC = 8.04  (Random initialization)
Step 5K:    BPC = 2.23  (Initial curriculum complete)
Step 10K:   BPC = 1.98  (Mid-training)
Step 20K:   BPC = 1.85  (Final)
```

**Improvement:** **6.19 BPC reduction** (77% improvement)

## Critical Fix: AMP Stability

**Problem:** Float16 overflow in Hebbian Memory with low plasticity (α=0.1)  
**Solution:** Force float32 computation for memory module

```python
@torch.amp.autocast('cuda', enabled=False)
def forward(self, x):
    x = x.float()  # Bypass AMP for numerical stability
    # ... Hebbian computation ...
    return out.to(input_dtype)
```

This fix enables stable 20K+ step training with AMP enabled.

## Documentation

- [Architecture Guide](docs/architecture.md) - Technical deep dive
- [Training Guide](docs/training.md) - Training from scratch
- [Inference Guide](docs/inference.md) - Generation and sampling
- [API Reference](docs/api.md) - Code documentation
- [RFC 007: Curriculum Learning](docs/RFC_007_Curriculum_Learning.md) - Phase 7 design

## Model Files

- `best_model_curriculum.pth` - Best checkpoint (Val BPC: 1.78)
- `last_model_curriculum.pth` - Final model state (20K steps)
- `metrics_curriculum.json` - Full training metrics

## Next Steps

### Recommended Improvements

1. **Extended Training:** 30K-50K steps for further convergence
2. **Larger Model:** Increase d_model=768, n_layers=8
3. **Longer Context:** Extend to 2048 token window
4. **Fine-tuning:** Domain-specific Turkish datasets

### Research Directions

- Adaptive plasticity scheduling
- Multi-stage curriculum optimization
- Cross-lingual transfer learning
- Sparse Hebbian memory

## Citation

```bibtex
@software{agiformer2025,
  title={AGIFORMER: Byte-Level Language Model with Hebbian Memory and Neuroplasticity},
  author={inkbytefo},
  year={2025},
  note={Phase 7: Curriculum Learning with Dynamic Plasticity},
  url={https://github.com/inkbytefo/agi-former}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch
- Turkish Wikipedia dataset (trwiki)
- Turkish Dictionary dataset (TDK)
- Inspired by Fast Weights, Linear Transformers, and developmental neuroscience

---

**Developer:** inkbytefo  
**Phase:** 7 (Curriculum Learning & Neuroplasticity)  
**Status:** Production Ready ✅  
**Last Updated:** 2025-11-23
