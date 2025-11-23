## Developer: inkbytefo
## Modified: 2025-11-23

import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory import HebbianMemory

class SlidingWindowAttention(nn.Module):
    """
    Local Attention mechanism restricted to a sliding window.
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

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class HybridBlock(nn.Module):
    """
    Combines Sliding Window Attention (Local) and Hebbian Memory (Global).
    
    v2.0 UPGRADE:
    - Gated Fusion: Output = sigmoid(g)*Local + (1-g)*Global
    - SwiGLU: Better activation function
    - RMSNorm: Better stability
    """
    def __init__(self, d_model, num_heads, window_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model) # Keeping LayerNorm for consistency with Encoder/Memory for now
        
        # Local Precision
        self.attn = SlidingWindowAttention(d_model, num_heads, window_size)
        
        # Global Context (Hebbian Memory)
        self.memory = HebbianMemory(d_model, num_heads, dropout)
        
        # v2.0: Gated Fusion
        # Learned gate to balance Local vs Global
        self.fusion_gate = nn.Linear(d_model, 1)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        # v2.0: SwiGLU MLP
        self.mlp = SwiGLU(d_model, 4 * d_model, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        
        # Parallel Branches
        attn_out = self.attn(x_norm)
        memory_out = self.memory(x_norm)
        
        # v2.0: Gated Fusion
        # g = sigmoid(W * x)
        # If g -> 1, prefer Local Attention
        # If g -> 0, prefer Global Memory
        g = torch.sigmoid(self.fusion_gate(x_norm))
        
        combined = (g * attn_out) + ((1 - g) * memory_out)
        
        # Fusion
        x = residual + self.out_proj(combined)
        
        # MLP
        x = x + self.mlp(self.norm_mlp(x))
        
        return x
