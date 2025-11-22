## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
from typing import Optional
from .encoder import ByteLatentEncoder
from .layers import HybridBlock

class LocalAutoregressiveHead(nn.Module):
    """
    Latent vector -> Bytes (Autoregressive).
    Global Model -> Latent -> Local Model -> Bytes
    """
    def __init__(self, d_model, patch_size, hidden_dim=256):
        super().__init__()
        self.patch_size = patch_size
        
        # Project latent to be the initial state or context
        self.proj_latent = nn.Linear(d_model, hidden_dim)
        
        # Byte embedding for the local decoder
        self.byte_emb = nn.Embedding(256, hidden_dim)
        
class AGIFORMER(nn.Module):
    """
    AGIFORMER: A Byte-Latent Hybrid Architecture.
    """
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        num_heads: int = 8,
        patch_size: int = 4,
        window_size: int = 128,
        vocab_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = ByteLatentEncoder(
            d_model=d_model,
            patch_size=patch_size,
            dropout=dropout
        )
        
        self.layers = nn.ModuleList([
            HybridBlock(
                d_model=d_model,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # 3. Head (Local Autoregressive)
        logits = self.head(x, target_bytes)
        
        return logits
