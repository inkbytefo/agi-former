## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory import HebbianMemory  # NEW IMPORT

class SlidingWindowAttention(nn.Module):
    """
    Local Attention mechanism restricted to a sliding window.
    Using standard SDPA for stability.
    """
    def __init__(self, d_model: int, num_heads: int, window_size: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        E = self.head_dim
        scale = 1.0 / (E ** 0.5)
        
        qkv = self.qkv(x).reshape(B, L, 3, H, E).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Manual Attention for Stability
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Construct Mask
        ones = torch.ones(L, L, device=x.device, dtype=torch.bool)
        causal_mask = ones.triu(1)
        window_mask = ones.tril(-self.window_size)
        mask = causal_mask | window_mask
        
        scores = scores.masked_fill(mask, -1e4)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

class HybridBlock(nn.Module):
    """
    Combines Sliding Window Attention (Local) and Hebbian Memory (Global).
    """
    def __init__(self, d_model, num_heads, window_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        
        # Local Precision
        self.attn = SlidingWindowAttention(d_model, num_heads, window_size)
        
        # Global Context (Hebbian Memory) - Replaces LinearAttention
        self.memory = HebbianMemory(d_model, num_heads, dropout)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm_mlp = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        
        # Parallel Branches
        attn_out = self.attn(x_norm)
        memory_out = self.memory(x_norm) # Using Hebbian Memory
        
        # Fusion
        x = residual + self.out_proj(attn_out + memory_out)
        
        # MLP
        x = x + self.mlp(self.norm_mlp(x))
        
        return x
