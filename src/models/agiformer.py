## Developer: inkbytefo
## Modified: 2025-11-23

import torch
import torch.nn as nn
from typing import Optional
from .encoder import ByteLatentEncoder
from .layers import HybridBlock
from .reasoning import RecurrentReasoningBlock

class LocalAutoregressiveHead(nn.Module):
    """
    Decodes latent vectors back into bytes.
    
    v2.0 UPGRADE:
    - Replaced GRU with Parallel MLP Decoder.
    - Predicts all 4 bytes in the patch simultaneously.
    - Faster and avoids sequential bottleneck.
    """
    def __init__(self, d_model, patch_size, hidden_dim=512):
        super().__init__()
        self.patch_size = patch_size
        
        # MLP Decoder
        # Input: Latent (D)
        # Output: Patch_Size * 256 (Logits for each byte in patch)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_size * 256)
        )

    def forward(self, latents, target_bytes=None, temperature=0.0):
        B, N, D = latents.shape
        
        # Predict all bytes at once
        logits = self.decoder(latents) # (B, N, P*256)
        logits = logits.view(B, N, self.patch_size, 256)
        
        return logits

    def _inference(self, latents, latent_context, temperature):
        # Deprecated in v2.0 - forward handles everything
        pass

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
    ):
        super().__init__()
        
        self.encoder = ByteLatentEncoder(d_model, patch_size, dropout)
        
        # Hybrid Blocks now use Hebbian Memory
        self.layers = nn.ModuleList([
            HybridBlock(d_model, num_heads, window_size, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.reasoning = RecurrentReasoningBlock(d_model, thinking_steps, dropout)
        self.head = LocalAutoregressiveHead(d_model, patch_size)

    def forward(self, x, target_bytes=None, temperature=0.0):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        x = self.reasoning(x)
        logits = self.head(x, target_bytes, temperature=temperature)
        return logits
