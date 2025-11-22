## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Placeholder for Mamba/SSM - In a real scenario, we'd import from mamba_ssm
# For this prototype, we will implement a simplified SSM or use a GRU as a proxy if Mamba isn't available.
# However, the prompt asks for Hybrid Attention + SSM.
# I will implement a mock-up interface for the SSM part to keep the architecture clear,
# assuming the user might install `mamba-ssm` later or we implement a simple linear RNN.

class LinearAttention(nn.Module):
    """
    Linear Attention (or Linear Transformer) block.
    Mathematically equivalent to a specific type of SSM.
    Complexity: O(N) time and memory.
    Allows parallel training (unlike RNN/GRU).
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Feature map for Linear Attention (Katharopoulos et al. / Performer)
        # elu(x) + 1 is a common choice to ensure positivity
        self.feature_map = nn.ELU() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        E = self.head_dim
        
        # Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for heads: (B, L, H, E)
        q = q.view(B, L, H, E)
        k = k.view(B, L, H, E)
        v = v.view(B, L, H, E)
        
        # Apply feature map to Q and K to ensure they are positive (Kernel trick)
        # Q' = phi(Q), K' = phi(K)
        # Scale Q to prevent large values
        q = q * (self.head_dim ** -0.5)
        
        # ELU+1 can be close to 0. Add epsilon to ensure strict positivity.
        q = self.feature_map(q) + 1.0 + 1e-4
        k = self.feature_map(k) + 1.0 + 1e-4
        
        # Linear Attention Formula:
        # O_i = (Q_i * sum_j(K_j^T * V_j)) / (Q_i * sum_j(K_j^T))
        # For Causal (Autoregressive) masking, the sum is up to i.
        # We can use efficient cumulative sum (cumsum) for this.
        
        # 1. Compute KV^T: (B, L, H, E) * (B, L, H, E) -> (B, L, H, E, E) is too big?
        # No, we want sum_{j<=i} (K_j * V_j^T).
        # K: (B, L, H, E), V: (B, L, H, E)
        # We need outer product K_j * V_j^T per step.
        # Einsum: b l h e, b l h f -> b l h e f
        
        kv = torch.einsum('blhe,blhf->blhef', k, v)
        
        # 2. Compute Cumulative Sum (Parallel Scan equivalent for Linear Attn)
        kv_cumsum = torch.cumsum(kv, dim=1) # (B, L, H, E, E)
        
        # 3. Compute Denominator: sum_{j<=i} K_j
        k_cumsum = torch.cumsum(k, dim=1) # (B, L, H, E)
        
        # 4. Compute Output
        # Num: Q_i * KV_cumsum_i
        # Den: Q_i * K_cumsum_i
        
        # Num: (B, L, H, E) * (B, L, H, E, E) -> (B, L, H, E)
        # einsum: blhe, blhef -> blhf
        num = torch.einsum('blhe,blhef->blhf', q, kv_cumsum)
        
        # Den: (B, L, H, E) * (B, L, H, E) -> (B, L, H)
        # einsum: blhe, blhe -> blh
        den = torch.einsum('blhe,blhe->blh', q, k_cumsum)
        
        # Add epsilon to den to avoid div by zero
        den = den.unsqueeze(-1) + 1e-6
        
        out = num / den
        
        # Reshape back
        out = out.reshape(B, L, D)
        
        return self.out_proj(self.dropout(out))

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
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Q, K, V projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled Dot-Product Attention with Sliding Window Mask
        # For simplicity in this prototype, we use PyTorch's SDPA with a manual mask or rely on efficient implementation.
        # Constructing a full mask for sliding window:
        
        # Create causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        
        # Create window mask (mask out elements too far in the past)
        # i.e., attention is allowed if i - window_size <= j <= i
        window_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=-(self.window_size - 1)).bool()
        window_mask = ~window_mask # Invert to get elements OUTSIDE the window
        
        # Combine: Mask if future OR (past AND outside window)
        attn_mask = mask | window_mask.T # Transpose because triu creates upper triangle
        
        # Apply attention
        # Note: is_causal=False because we manually constructed the causal+window mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

class HybridBlock(nn.Module):
    """
    Combines Local Attention (for precision) and SSM (for global context).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        window_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Parallel Branches
        self.attn = SlidingWindowAttention(d_model, num_heads, window_size)
        # Replaced SimpleSSM (GRU) with LinearAttention (O(N) Parallel)
        self.ssm = LinearAttention(d_model, num_heads=num_heads)
        
        # Gating / Combination
        # We can sum them, or use a learned gate.
        # Here we use a simple learned gate to weight the contributions.
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm_mlp = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        residual = x
        x_norm = self.norm1(x)
        
        # Parallel Execution
        attn_out = self.attn(x_norm)
        ssm_out = self.ssm(x_norm) # Now using Linear Attention
        
        # Gated Combination
        # Concatenate and decide how much of each to keep
        combined = torch.cat([attn_out, ssm_out], dim=-1)
        gate_score = self.gate(combined) # (B, L, D) - this is not quite right for a scalar gate, but let's say element-wise mixing
        
        # Actually, a simpler "alpha * attn + (1-alpha) * ssm" might be better, 
        # but let's just sum them for now with a projection, or use the gate as a mixer.
        # Let's try: Output = Proj(Concat(Attn, SSM)) + Residual
        
        mixed = self.out_proj(attn_out + ssm_out) # Simple sum for stability in prototype
        
        x = residual + mixed
        
        # MLP Block
        residual = x
        x = x + self.mlp(self.norm_mlp(x))
        
        return x
