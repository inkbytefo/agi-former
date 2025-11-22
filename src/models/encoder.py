## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteLatentEncoder(nn.Module):
    """
    Encodes raw byte sequences into latent patch representations.
    
    This module replaces traditional tokenizers by learning to compress
    raw bytes directly into a higher-dimensional latent space.
    """
    def __init__(
        self,
        d_model: int,
        patch_size: int = 4,
        dropout: float = 0.1,
        max_len: int = 4096
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        
        # Byte Embedding: 256 possible byte values -> d_model
        self.byte_embedding = nn.Embedding(256, d_model)
        
        # Patching mechanism: Strided Convolution
        self.patch_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        
        # RoPE (Rotary Positional Embeddings)
        # We precompute frequencies for RoPE
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        
        # Create position indices
        t = torch.arange(N, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq) # (N, D/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (N, D)
        
        # Apply rotation
        # Simple implementation: x_rotated = x * cos(emb) + rotate_half(x) * sin(emb)
        # rotate_half: [-x2, x1, -x4, x3, ...]
        
        x1 = x[..., :D//2]
        x2 = x[..., D//2:]
        rotate_half_x = torch.cat((-x2, x1), dim=-1)
        
        return x * emb.cos() + rotate_half_x * emb.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq_Len) tensor of uint8 bytes (0-255)
            
        Returns:
            latents: (Batch, Seq_Len // patch_size, d_model)
        """
        # 1. Embed bytes
        x = self.byte_embedding(x.long())
        
        # 2. Transpose for Conv1d
        x = x.transpose(1, 2)
        
        # 3. Apply Patching
        x = self.patch_conv(x)
        
        # 4. Transpose back
        x = x.transpose(1, 2)
        
        # 5. Apply RoPE
        x = self.apply_rope(x)
        
        # 6. Normalize and Dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x
