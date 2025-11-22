# API Reference

## Module: `src.models.encoder`

### Class: `ByteLatentEncoder`

Converts byte sequences into latent patches with positional embeddings.

```python
class ByteLatentEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 4,
        dropout: float = 0.1
    )
```

**Parameters:**
- `d_model` (int): Latent dimension size
- `patch_size` (int): Number of bytes per patch
- `dropout` (float): Dropout probability

**Methods:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (Batch, Seq_Len) - Input bytes [0-255]
    
    Returns:
        (Batch, Num_Patches, d_model) - Latent patches
    """
```

---

## Module: `src.models.layers`

### Class: `LinearAttention`

$O(N)$ causal attention using ELU feature maps.

```python
class LinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    )
```

**Methods:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (Batch, Seq_Len, d_model)
    
    Returns:
        (Batch, Seq_Len, d_model)
    """
```

**Algorithm:**
```
Q, K, V = elu(Wq x) + 1, elu(Wk x) + 1, Wv x
Attention = (Q @ cumsum(K ⊗ V)) / (Q @ cumsum(K) + ε)
```

---

### Class: `SlidingWindowAttention`

Causal attention with fixed window size.

```python
class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int
    )
```

**Parameters:**
- `window_size` (int): Maximum distance for attention (default: 128)

---

### Class: `HybridBlock`

Combines LinearAttention + SlidingWindowAttention in parallel.

```python
class HybridBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        dropout: float
    )
```

**Methods:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (Batch, Seq_Len, d_model)
    
    Returns:
        (Batch, Seq_Len, d_model)
    
    Algorithm:
        attn_out = SlidingWindowAttention(norm(x))
        ssm_out = LinearAttention(norm(x))
        x = x + out_proj(attn_out + ssm_out)
        x = x + MLP(norm(x))
    """
```

---

## Module: `src.models.reasoning`

### Class: `RecurrentReasoningBlock`

System 2 thinking loop with gated residual updates.

```python
class RecurrentReasoningBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        thinking_steps: int = 3,
        dropout: float = 0.1
    )
```

**Parameters:**
- `thinking_steps` (int): Number of refinement iterations

**Methods:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (Batch, Seq_Len, d_model) - Initial latent
    
    Returns:
        (Batch, Seq_Len, d_model) - Refined latent
    
    Algorithm:
        for t in range(thinking_steps):
            update = MLP(norm(x))
            gate = sigmoid(W_gate @ norm(x))
            x = x + gate * update
    """
```

---

## Module: `src.models.agiformer`

### Class: `LocalAutoregressiveHead`

GRU-based byte decoder with teacher forcing.

```python
class LocalAutoregressiveHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        hidden_dim: int = 256
    )
```

**Methods:**
```python
def forward(
    self,
    latents: torch.Tensor,
    target_bytes: Optional[torch.Tensor] = None,
    temperature: float = 0.0
) -> torch.Tensor:
    """
    Args:
        latents: (Batch, Num_Patches, d_model)
        target_bytes: (Batch, Num_Patches * patch_size) - For training
        temperature: Sampling temperature (0 = greedy)
    
    Returns:
        Training: (Batch, Num_Patches, patch_size, 256) - Logits
        Inference: (Batch, Num_Patches, patch_size) - Byte IDs
    """
```

---

### Class: `AGIFORMER`

Main model class.

```python
class AGIFORMER(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        num_heads: int = 8,
        patch_size: int = 4,
        window_size: int = 128,
        vocab_size: int = 256,
        dropout: float = 0.1,
        thinking_steps: int = 3
    )
```

**Parameters:**
- `d_model`: Latent dimension
- `n_layers`: Number of HybridBlocks
- `num_heads`: Attention heads per layer
- `patch_size`: Bytes per patch
- `window_size`: Local attention window
- `vocab_size`: Always 256 (bytes)
- `dropout`: Dropout probability
- `thinking_steps`: System 2 iterations

**Methods:**
```python
def forward(
    self,
    x: torch.Tensor,
    target_bytes: Optional[torch.Tensor] = None,
    temperature: float = 0.0
) -> torch.Tensor:
    """
    Full forward pass: Encoder → Backbone → Reasoning → Decoder
    
    Args:
        x: (Batch, Seq_Len) - Input bytes
        target_bytes: (Batch, Seq_Len_Target) - For training
        temperature: Sampling temperature
    
    Returns:
        Training: (Batch, Num_Patches, patch_size, 256)
        Inference: (Batch, Num_Patches, patch_size)
    """
```

---

## Module: `src.data.real_data`

### Class: `Enwik8Dataset`

PyTorch dataset for enwik8.

```python
class Enwik8Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str = "./data",
        split: str = "train",
        seq_len: int = 1024
    )
```

**Parameters:**
- `split`: "train", "val", or "test"
- `seq_len`: Sequence length per sample

**Methods:**
```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        input: (seq_len,) - Context bytes
        target: (seq_len,) - Next-patch bytes
    """
```

### Function: `get_enwik8_dataloader`

Creates DataLoader with automatic download.

```python
def get_enwik8_dataloader(
    batch_size: int,
    seq_len: int,
    split: str = "train"
) -> torch.utils.data.DataLoader:
    """
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        split: "train", "val", or "test"
    
    Returns:
        DataLoader yielding (input, target) batches
    """
```

---

## Utility Scripts

### `train.py`

Main training loop.

**Key Functions:**
```python
def train_step(model, batch, optimizer, criterion):
    """Single training step"""
    
def validate(model, val_loader, criterion):
    """Validation loop"""
```

### `generate.py`

Inference with temperature sampling.

**Key Function:**
```python
def generate_text(
    model_path: str,
    prompt_text: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7
) -> None:
    """Generate text from prompt"""
```

### `inspect_reasoning.py`

System 2 diagnostics.

**Key Function:**
```python
def inspect_system_2(model_path: str) -> None:
    """
    Measures:
    - Latent refinement (Δz)
    - Gate biases
    - Parameter health
    """
```

---

## Example Usage

### Training from Scratch
```python
from src.models.agiformer import AGIFORMER
from src.data.real_data import get_enwik8_dataloader
import torch.optim as optim

model = AGIFORMER(d_model=512, n_layers=6, thinking_steps=3)
train_loader = get_enwik8_dataloader(batch_size=4, seq_len=1024)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

for batch in train_loader:
    x, target = batch
    logits = model(x, target_bytes=target)
    loss = F.cross_entropy(logits.view(-1, 256), target.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
```

### Custom Inference
```python
model = AGIFORMER()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

prompt_bytes = torch.tensor([ord(c) for c in "Hello world"])
with torch.no_grad():
    output = model(prompt_bytes.unsqueeze(0), temperature=0.7)

generated = output[0, -1, :].tolist()
text = ''.join([chr(b) for b in generated if 32 <= b <= 126])
print(text)
```

---

## Type Hints Summary

```python
# Common types
Tensor = torch.Tensor
IntTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

# Shapes (notation)
B = Batch size
L = Sequence length
N = Number of patches (L / patch_size)
P = Patch size
D = d_model
H = num_heads
V = Vocabulary size (256)

# Input/Output shapes
Input: (B, L) IntTensor
Latent: (B, N, D) FloatTensor
Logits: (B, N, P, V) FloatTensor
Output: (B, N, P) IntTensor
```
