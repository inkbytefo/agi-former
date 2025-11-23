# AGIFORMER: Byte-Level Language Model with Neuroplasticity

> **Status:** Phase 7 - Curriculum Learning ✅ **Complete**  
> **Latest Achievement:** 20K curriculum training with 77% BPC reduction

A research implementation of a byte-level language model featuring:
- 🧠 **Hebbian Memory** with dynamic neuroplasticity
- 📚 **Curriculum Learning** (3-stage developmental approach)
- 🔄 **System 2 Reasoning** (iterative thinking loop)
- 🚀 **Linear Complexity** attention mechanism

## Quick Start

### Installation
```bash
pip install torch datasets tqdm
```

### Training (Curriculum Learning)
```bash
python train_curriculum.py  # 20K steps, 3 curriculum stages
```

### Inference
```bash
python generate.py best_model_curriculum.pth
```

### Testing
```bash
python test_recall.py best_model_curriculum.pth  # Memory test
python inspect_reasoning.py                        # System 2 diagnostics
```

## Architecture

```
Bytes → Encoder (RoPE) → Hebbian Memory → Reasoning Loop → Local RNN → Bytes
         (Patches)        (Dynamic λ)       (3 steps)        (Autoregressive)
```

### Core Components

- **ByteLatentEncoder:** Patches bytes into latent vectors with RoPE
- **HebbianMemory:** Fast weights with learnable decay + neuroplasticity (α)
- **RecurrentReasoningBlock:** 3-step iterative thinking loop (System 2)
- **LocalAutoregressiveHead:** GRU-based byte decoder

See [docs/architecture.md](docs/architecture.md) for technical details.

## Features

✅ **No Tokenization** - Universal byte-level processing  
✅ **Linear Complexity** - O(N) attention with Hebbian memory  
✅ **Neuroplasticity** - Dynamic memory consolidation (α: 0.1 → 0.99)  
✅ **Curriculum Learning** - 3-stage developmental training  
✅ **Active Reasoning** - Verified thinking loop (Δz = 12.7)  
✅ **AMP Compatible** - Mixed precision training with stability fixes  

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
