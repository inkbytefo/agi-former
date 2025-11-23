## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteLatentEncoder(nn.Module):
    """
    Encodes raw byte sequences into latent patch representations.
    
    v2.0 UPGRADE:
    - Soft Patching: Kernel size 6, Stride 4 (Overlap of 2 bytes)
    - RMSNorm: Better stability
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
        
        # v2.0 CRITICAL FIX: Causal Patching (No Future Leakage)
        # Kernel=6, Stride=4 -> 2 bytes overlap between patches
        # IMPORTANT: Padding must be CAUSAL (left-only) to prevent seeing future bytes
        # We add manual left padding in forward() instead of Conv1d padding
        self.patch_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=6,  
            stride=patch_size,
            padding=0  # Changed from 1 to 0 - manual causal padding in forward
        )
        
        # RoPE (Rotary Positional Embeddings)
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        
        # v2.0: RMSNorm
        self.norm = nn.LayerNorm(d_model) # Keeping LayerNorm for now to minimize changes, or switch to RMSNorm if defined globally
        self.dropout = nn.Dropout(dropout)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        
        # Create position indices
        t = torch.arange(N, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq) # (N, D/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (N, D)
        
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
        
        # 2. Transpose for Conv1d: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        
        # 3. CAUSAL PADDING (Left-only, no future leakage)
        # Kernel=6, Stride=4. To prevent model from seeing future bytes,
        # we pad only on the LEFT (past) side.
        # Padding size = kernel_size - stride = 6 - 4 = 2
        x = torch.nn.functional.pad(x, (2, 0))  # (left, right)
        
        # 4. Apply Patching
        x = self.patch_conv(x)
        
        # 4. Transpose back
        x = x.transpose(1, 2)
        
        # 5. Apply RoPE
        x = self.apply_rope(x)
        
        # 6. Normalize and Dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x
