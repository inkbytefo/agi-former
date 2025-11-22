# AGIFORMER

**Byte-Level Language Model with System 2 Reasoning**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

AGIFORMER is a novel language model architecture that combines:
- **Byte-level processing** (no tokenization)
- **Linear Attention** for $O(N)$ complexity
- **System 2 Reasoning** via iterative latent refinement

**Key Results:**
- Final BPC: **2.2578** on enwik8
- System 2 active thinking: **Δz = 12.7**
- Zero NaN crashes across training
- 15 min training time (5000 steps on T4)

## Quick Start

### Installation
```bash
git clone https://github.com/inkbytefo/agi-former.git
cd agi-former
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Inference
```bash
python generate.py
```

### System 2 Diagnostics
```bash
python inspect_reasoning.py
```

## Architecture

```
Bytes → Encoder (RoPE) → Linear Attention → Reasoning Loop → Local RNN → Bytes
         (Patches)        (Global Context)    (3 steps)     (Autoregressive)
```

### Components
- **ByteLatentEncoder:** Patches bytes into latent vectors with RoPE
- **LinearAttention:** $O(N)$ causal attention with ELU feature maps
- **RecurrentReasoningBlock:** 3-step iterative thinking loop (System 2)
- **LocalAutoregressiveHead:** GRU-based byte decoder

See [docs/architecture.md](docs/architecture.md) for details.

## Features

✅ **No Tokenization** - Universal byte-level processing  
✅ **Linear Complexity** - Scales to long contexts  
✅ **Active Reasoning** - Verified thinking loop (Δz = 12.7)  
✅ **Stable Training** - No NaN, robust gradient flow  
✅ **Temperature Sampling** - Diverse inference outputs  

## Documentation

- [Architecture Guide](docs/architecture.md) - Technical deep dive
- [Training Guide](docs/training.md) - How to train from scratch
- [Inference Guide](docs/inference.md) - Generation and sampling
- [API Reference](docs/api.md) - Code documentation

## Results

### Quantitative
- **BPC:** 2.2578 (enwik8, 5000 steps)
- **Training Time:** 15 minutes (T4 GPU)
- **Stability:** 0 NaN occurrences

### Qualitative
```
Prompt: "The history of "
Output: "Tomadination of the [[New Gouple de aparty]] with the June 
         competition became at the..."
```

- Wikipedia syntax learned (`[[...]]`)
- Clause structure emerging
- "Thinking pause" (whitespace before output)

## Citation

```bibtex
@software{agiformer2025,
  title={AGIFORMER: Byte-Level Language Model with System 2 Reasoning},
  author={inkbytefo},
  year={2025},
  url={https://github.com/inkbytefo/agi-former}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch
- Trained on enwik8 dataset
- Inspired by Linear Transformers and System 2 reasoning research

---

**Developer:** inkbytefo  
**Contact:** [GitHub](https://github.com/inkbytefo)  
**Status:** Proof of Concept - Complete ✅
