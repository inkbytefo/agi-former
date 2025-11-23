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
        # PHASE 8 OPTIMIZATION: Start with high retention (sticky memory)
        # sigmoid(8.0) ≈ 0.9997 → very strong initial retention
        # Still learnable, model can adjust down if needed
        self.decay_logits = nn.Parameter(torch.tensor([8.0] * num_heads)) 
        
        self.norm = nn.LayerNorm(d_model)
        
        # Plasticity Factor (Alpha) - Controlled externally
        self.plasticity = 1.0

    def set_plasticity(self, alpha):
        """
        Updates the plasticity coefficient (alpha).
        alpha: float in [0, 1]. 
               0.1 -> Childhood (Fast forgetting)
               0.99 -> Adulthood (Stable memory)
        """
        self.plasticity = alpha

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        # CRITICAL: Bypass AMP for this entire module to prevent NaN
        # With plasticity=0.1, decay factors become exp(±50) and the cumsum
        # operations accumulate massive intermediate values that overflow in float16
        # We must use float32 for all computations including linear layers
        x = x.float()  # Ensure input is float32
        input_dtype = x.dtype
        
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
        
        # 3. Decay Factor (Lambda) - PHASE 8 OPTIMIZATION
        # Map sigmoid (0,1) to (0.995, 1.0) for stronger retention
        # 0.995^1024 = 0.006 (still remembers 0.6% after 1024 steps)
        # vs 0.99^1024 = 0.00004 (essentially forgotten)
        raw_sigmoid = torch.sigmoid(self.decay_logits).view(1, 1, H, 1)
        lambdas = 0.995 + (0.005 * raw_sigmoid)  # Tighter, stickier range
        
        # Apply Plasticity Schedule
        # Effective Lambda = Lambda * Alpha
        # If Alpha is low (childhood), decay is very fast.
        lambdas = lambdas * self.plasticity
        
        # 4. Parallel Hebbian Update
        # Formula: O_i = (Q_i * sum_{j=1}^i lambda^{i-j} K_j^T V_j)
        # Implementation: Q_i * lambda^i * cumsum(lambda^-j * K_j * V_j)
        
        indices = torch.arange(L, device=x.device, dtype=torch.float32).view(1, L, 1, 1)
        
        # Use log-space arithmetic to prevent overflow/underflow
        log_lambdas = torch.log(lambdas.clamp(min=1e-10))
        
        # Clamp the exponent BEFORE exp() to prevent overflow
        # We use ±50 as a safe range that works for float32
        exp_k = (-indices * log_lambdas).clamp(min=-50, max=50)
        exp_q = (indices * log_lambdas).clamp(min=-50, max=50)
        
        # Compute decay factors
        decay_k = torch.exp(exp_k)  # lambda^-indices
        decay_q = torch.exp(exp_q)  # lambda^indices
        
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
        
        # Convert back to input dtype before applying norm and dropout
        out = self.dropout(self.norm(out))
        return out.to(input_dtype)  # Convert back to original dtype

