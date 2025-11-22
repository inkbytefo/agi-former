## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
import torch.nn.functional as F
## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    """
    Numerically Stable Linear Attention.
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Output normalization to prevent residual explosion
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        E = self.head_dim
        
        # Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape: (B, L, H, E)
        q = q.view(B, L, H, E)
        k = k.view(B, L, H, E)
        v = v.view(B, L, H, E)
        
        # Feature Map: ELU + 1 (Standard)
        # Stability Fix: Normalize Q and K to keep dot products in check
        # Stability Fix 2: Ensure strictly positive to avoid 0 denominator
        q = F.elu(q) + 1.0 + 1e-4
        k = F.elu(k) + 1.0 + 1e-4
        
        # Scale to prevent huge sums
        # Standard attention divides by sqrt(dk), here we do it to Q
        q = q / torch.sqrt(torch.tensor(E, dtype=q.dtype))
        
        # Linear Attention Core (O(N))
        # sum_{j<=i} (K_j * V_j^T)
        
        # 1. KV Calculation (No huge expansion, einsum handles dimensions)
        # (B, L, H, E) * (B, L, H, E) -> (B, L, H, E, E)
        kv = torch.einsum('blhe,blhf->blhef', k, v)
        
        # 2. Cumsum (Parallel Scan)
        kv_cumsum = torch.cumsum(kv, dim=1) 
        
        # 3. K Cumsum (Denominator)
        k_cumsum = torch.cumsum(k, dim=1) 
        
        # 4. Numerator: Q * KV_cumsum
        # (B, L, H, E) * (B, L, H, E, E) -> (B, L, H, E)
        num = torch.einsum('blhe,blhef->blhf', q, kv_cumsum)
        
        # 5. Denominator: Q * K_cumsum
        # (B, L, H, E) * (B, L, H, E) -> (B, L, H)
        den = torch.einsum('blhe,blhe->blh', q, k_cumsum)
        
        # Stability Fix: Larger epsilon and absolute check
        den = den.unsqueeze(-1) + 1e-4
        
        out = num / den
        
        # Reshape and Project
        out = out.reshape(B, L, D)
        out = self.out_proj(out)
        
        # Final Norm/Dropout
        return self.dropout(self.norm(out))

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
        
        # (B, L, 3, H, E) -> (3, B, H, L, E)
        qkv = self.qkv(x).reshape(B, L, 3, H, E).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Manual Attention for Stability
        # (B, H, L, E) @ (B, H, E, L) -> (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Construct Mask
        # Window constraint: j > i - window_size  =>  i - j < window_size
        # Causal: j <= i
        # Valid: i - window_size < j <= i
        
        # Mask is True where we want to BLOCK
        # 1. Causal Block: j > i (triu(1))
        # 2. Window Block: j <= i - window_size (tril(-window_size))
        
        ones = torch.ones(L, L, device=x.device, dtype=torch.bool)
        causal_mask = ones.triu(1)
        window_mask = ones.tril(-self.window_size)
        
        mask = causal_mask | window_mask
        
        # Apply Mask
        # Use -1e4 instead of -inf for stability
        scores = scores.masked_fill(mask, -1e4)
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        
        # Dropout (if needed, but not in init args currently)
        # attn = F.dropout(attn, p=0.1) 
        
        # (B, H, L, L) @ (B, H, L, E) -> (B, H, L, E)
        out = torch.matmul(attn, v)
        
        # (B, H, L, E) -> (B, L, H, E) -> (B, L, D)
        out = out.transpose(1, 2).reshape(B, L, D)
        
        return self.proj(out)

class HybridBlock(nn.Module):
    def __init__(self, d_model, num_heads, window_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        
        self.attn = SlidingWindowAttention(d_model, num_heads, window_size)
        self.ssm = LinearAttention(d_model, num_heads, dropout)
        
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
        
        attn_out = self.attn(x_norm)
        if torch.isnan(attn_out).any():
            print("DEBUG: NaN detected in SlidingWindowAttention!")
            
        ssm_out = self.ssm(x_norm)
        if torch.isnan(ssm_out).any():
            print("DEBUG: NaN detected in LinearAttention (SSM)!")
        
        # Summation fusion
        x = residual + self.out_proj(attn_out + ssm_out)
        
        # MLP
        x = x + self.mlp(self.norm_mlp(x))
        
        return x
