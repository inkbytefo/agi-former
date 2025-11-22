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
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Construct Sliding Window Mask manually to avoid SDPA kernel issues with complex constraints
        # Or simply rely on PyTorch 2.0+ causal masking if strict window is hard
        # Stability Fix: Use a simpler causal mask + manual zeroing for window
        
        # Full Causal Mask
        mask = torch.ones(L, L, device=x.device, dtype=torch.bool).tril(0)
        # Window constraint: Keep only if i - j < window_size
        # i.e., j > i - window_size
        window_mask = torch.ones(L, L, device=x.device, dtype=torch.bool).tril(0).triu(-(self.window_size - 1))
        
        # Combine: We need a bias mask where False -> -inf
        # SDPA expects attn_mask to be float (0 or -inf) or bool (True=Masked/NotAllowed)
        # Let's use bool: True means "Don't Attend"
        
        final_mask = ~window_mask 
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=final_mask, is_causal=False)
        
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
        ssm_out = self.ssm(x_norm)
        
        # Summation fusion
        x = residual + self.out_proj(attn_out + ssm_out)
        
        # MLP
        x = x + self.mlp(self.norm_mlp(x))
        
        return x
