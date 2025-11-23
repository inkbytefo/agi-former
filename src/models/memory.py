## Developer: inkbytefo
## Modified: 2025-11-23

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale

class HebbianMemory(nn.Module):
    """
    Hebbian Memory Module v2.0 (Input-Dependent Decay).
    
    Implements the update rule:
    M_t = lambda_t * M_{t-1} + K_t * V_t^T
    
    UPGRADE v2.0:
    - Decay (lambda) is now a function of input x_t: lambda_t = sigmoid(W * x_t)
    - Allows the model to selectively forget or remember based on content.
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
        
        # v2.0: Input-Dependent Decay Network
        # Predicts a decay value for each head based on the input
        self.decay_net = nn.Linear(d_model, num_heads)
        
        # Initialize decay to be sticky (high retention) initially
        nn.init.constant_(self.decay_net.bias, 4.0) # sigmoid(4.0) ~= 0.98
        nn.init.normal_(self.decay_net.weight, std=0.02)
        
        self.norm = RMSNorm(d_model)
        
        # Plasticity Factor (Alpha) - Controlled externally
        self.plasticity = 1.0

    def set_plasticity(self, alpha):
        self.plasticity = alpha

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        # CRITICAL: Bypass AMP for this entire module to prevent NaN
        x = x.float()
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
        
        # Scale Q
        q = q / math.sqrt(E)
        
        # 3. Dynamic Decay (Lambda) - v2.0
        # lambda_t = sigmoid(decay_net(x_t))
        # Mapped to range [0.5, 1.0] to prevent total erasure, but allow fast decay
        decay_logits = self.decay_net(x) # (B, L, H)
        raw_sigmoid = torch.sigmoid(decay_logits).view(B, L, H, 1)
        lambdas = 0.9 + (0.1 * raw_sigmoid) # Range [0.9, 1.0] - Safer for gradients
        
        # Apply Plasticity
        lambdas = lambdas * self.plasticity
        
        # 4. Parallel Hebbian Update (Log-Space)
        # Since lambda varies per time step, we cannot use simple powers.
        # We must use cumulative sum of logs.
        # log_lambda_cum[t] = sum_{i=0}^t log(lambda_i)
        
        log_lambdas = torch.log(lambdas.clamp(min=1e-6)) # (B, L, H, 1)
        log_lambda_cum = torch.cumsum(log_lambdas, dim=1) # (B, L, H, 1)
        
        # Decay factors for Q and K
        # Q_t gets multiplied by lambda_t...lambda_L (future decay? No, standard form)
        # Standard parallel form:
        # O_t = sum_{j<=t} (prod_{k=j+1}^t lambda_k) K_j^T V_j
        # Log space: log_decay = log_cum[t] - log_cum[j]
        
        # To implement efficiently without O(L^2), we use the standard trick:
        # Multiply K by exp(-log_cum) and Q by exp(log_cum)
        
        # Clamp for stability
        log_lambda_cum = log_lambda_cum.clamp(min=-50, max=50)
        
        decay_k = torch.exp(-log_lambda_cum)
        decay_q = torch.exp(log_lambda_cum)
        
        k_decayed = k * decay_k
        q_decayed = q * decay_q
        
        # Memory State Accumulation (KV)
        kv = torch.einsum('blhe,blhf->blhef', k_decayed, v)
        memory_state = torch.cumsum(kv, dim=1)
        
        # Denominator (Z)
        k_sum_decayed = torch.cumsum(k_decayed, dim=1)
        
        # Read Operation
        # Num: (B, L, H, E) * (B, L, H, E, E) -> (B, L, H, E)
        num = torch.einsum('blhe,blhef->blhf', q_decayed, memory_state)
        
        # Den: (B, L, H, E) * (B, L, H, E) -> (B, L, H)
        den = torch.einsum('blhe,blhe->blh', q_decayed, k_sum_decayed)
        den = den.unsqueeze(-1) + 1e-6
        
        out = num / den
        
        # Final Projection
        out = out.reshape(B, L, D)
        out = self.out_proj(out)
        
        out = self.dropout(self.norm(out))
        return out.to(input_dtype)
