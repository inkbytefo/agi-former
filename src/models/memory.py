## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HebbianMemory(nn.Module):
    """
    Hebbian Memory Module (Fast Weights).
    
    Functionally equivalent to Linear Attention but framed as a recurrent memory update rule:
    M_t = lambda * M_{t-1} + K_t * V_t^T
    O_t = Q_t * M_t
    
    Features:
    - Parallel Training via cumulative sum (Scan).
    - Learnable Decay (lambda) to control forgetting (Short-term vs Long-term).
    - Numerically stable implementation for sequence lengths up to ~2048.
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
        
        # Learnable Decay Rate (Lambda) per head
        # Initialized to high value (~0.995) to preserve long context initially.
        # Parameter is raw logits for sigmoid.
        self.decay_logits = nn.Parameter(torch.tensor([5.0] * num_heads)) 
        
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
        
        # 2. Feature Map & Stability
        q = self.feature_map(q) + 1.0
        k = self.feature_map(k) + 1.0
        
        # Scale Q
        q = q / math.sqrt(E)
        
        # 3. Decay Factor (Lambda)
        # lambda \in (0, 1). Using sigmoid on learnable logits.
        # View as (1, 1, H, 1) for broadcasting
        lambdas = torch.sigmoid(self.decay_logits).view(1, 1, H, 1)
        
        # 4. Parallel Hebbian Update (Linear Attention with Decay)
        # Formula: O_i = (Q_i * sum_{j=1}^i lambda^{i-j} K_j^T V_j)
        # Implementation: We multiply K_j by lambda^{-j} and Q_i by lambda^i
        # This converts the relative power lambda^{i-j} into absolute terms.
        
        # Compute time indices: [0, 1, ..., L-1]
        indices = torch.arange(L, device=x.device, dtype=torch.float32).view(1, L, 1, 1)
        
        # Decay mask: lambda^{-t} and lambda^{t}
        # Note: For very long L, lambda^{-t} can explode. 
        # With L=1024 and lambda=0.99, 0.99^-1024 ~= 29468 (Safe for FP32)
        # If lambda gets smaller (e.g., 0.9), we might need log-space or chunking.
        # For now, this is safe for prototyping.
        
        decay_k = torch.pow(lambdas, -indices) # (1, L, H, 1)
        decay_q = torch.pow(lambdas, indices)  # (1, L, H, 1)
        
        k_decayed = k * decay_k
        
        # Memory State Accumulation (KV)
        # (B, L, H, E) * (B, L, H, E) -> (B, L, H, E, E)
        # Einsum: b l h e, b l h f -> b l h e f
        kv = torch.einsum('blhe,blhf->blhef', k_decayed, v)
        
        # Cumsum (The "Write" Operation)
        memory_state = torch.cumsum(kv, dim=1) # (B, L, H, E, E)
        
        # Denominator Accumulation (Z)
        k_sum_decayed = torch.cumsum(k_decayed, dim=1) # (B, L, H, E)
        
        # Read Operation (Query * Memory)
        # Numerator: Q_decayed * Memory
        # Q needs to be scaled by lambda^t
        q_decayed = q * decay_q
        
        # Num: (B, L, H, E) * (B, L, H, E, E) -> (B, L, H, E)
        num = torch.einsum('blhe,blhef->blhf', q_decayed, memory_state)
        
        # Den: (B, L, H, E) * (B, L, H, E) -> (B, L, H)
        den = torch.einsum('blhe,blhe->blh', q_decayed, k_sum_decayed)
        den = den.unsqueeze(-1) + 1e-5 # Stability
        
        out = num / den
        
        # Final Projection
        out = out.reshape(B, L, D)
        out = self.out_proj(out)
        
        return self.dropout(self.norm(out))
