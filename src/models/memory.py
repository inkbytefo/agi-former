## Developer: inkbytefo
## Modified: 2025-11-23

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HebbianMemory(nn.Module):
    """
    Hebbian Memory Module (Fast Weights).
    
    Implements the update rule:
    M_t = lambda * M_{t-1} + K_t * V_t^T
    O_t = Q_t * M_t
    
    CRITICAL CHANGE:
    To prevent numerical overflow in parallel computation (cumsum),
    the decay rate (lambda) is constrained to the range [0.99, 1.0].
    This ensures lambda^(-L) does not explode for L=1024.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feature map: ELU + 1 ensures positivity for valid probability kernel
        self.feature_map = nn.ELU()
        
        # Learnable Decay Parameter
        # Initialized to generate sigmoid output ~0.5, mapped to range later
        self.decay_logits = nn.Parameter(torch.zeros(num_heads)) 
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        E = self.head_dim
        
        # 1. Projections
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape (B, L, H, E)
        q = q.view(B, L, H, E)
        k = k.view(B, L, H, E)
        v = v.view(B, L, H, E)
        
        # 2. Feature Map (Kernel Trick)
        q = self.feature_map(q) + 1.0
        k = self.feature_map(k) + 1.0
        
        # Scale Q to prevent magnitude explosion
        q = q / math.sqrt(E)
        
        # 3. Decay Factor (Lambda) - STABILIZED
        # Map sigmoid (0,1) to (0.990, 1.0)
        # This prevents overflow. 0.99^-1024 ~= 29468 (Safe for FP32)
        raw_sigmoid = torch.sigmoid(self.decay_logits).view(1, 1, H, 1)
        lambdas = 0.99 + (0.01 * raw_sigmoid)
        
        # 4. Parallel Hebbian Update
        # Formula: O_i = (Q_i * sum_{j=1}^i lambda^{i-j} K_j^T V_j)
        # Implementation: Q_i * lambda^i * cumsum(lambda^-j * K_j * V_j)
        
        indices = torch.arange(L, device=x.device, dtype=torch.float32).view(1, L, 1, 1)
        
        decay_k = torch.pow(lambdas, -indices) # (1, L, H, 1)
        decay_q = torch.pow(lambdas, indices)  # (1, L, H, 1)
        
        k_decayed = k * decay_k
        
        # Memory State Accumulation (KV)
        # (B, L, H, E) * (B, L, H, E) -> (B, L, H, E, E)
        # Einsum: b l h e, b l h f -> b l h e f
        kv = torch.einsum('blhe,blhf->blhef', k_decayed, v)
        
        # Cumsum (The "Write" Operation)
        memory_state = torch.cumsum(kv, dim=1) # (B, L, H, E, E)
        
        # Denominator Accumulation (Z) for normalization
        k_sum_decayed = torch.cumsum(k_decayed, dim=1) # (B, L, H, E)
        
        # Read Operation (Query * Memory)
        q_decayed = q * decay_q
        
        # Num: (B, L, H, E) * (B, L, H, E, E) -> (B, L, H, E)
        num = torch.einsum('blhe,blhef->blhf', q_decayed, memory_state)
        
        # Den: (B, L, H, E) * (B, L, H, E) -> (B, L, H)
        den = torch.einsum('blhe,blhe->blh', q_decayed, k_sum_decayed)
        den = den.unsqueeze(-1) + 1e-6 # Stability epsilon
        
        out = num / den
        
        # Final Projection
        out = out.reshape(B, L, D)
        out = self.out_proj(out)
        
        return self.dropout(self.norm(out))
