## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn

class RecurrentReasoningBlock(nn.Module):
    """
    System 2 Thinking Module.
    Refines the latent representation through N steps of recurrence.
    Formula: z_{t+1} = z_t + MLP(LayerNorm(z_t))
    """
    def __init__(self, d_model, thinking_steps=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.thinking_steps = thinking_steps
        
        # The "Thinking" Core
        # A dense MLP that transforms the latent space
        self.think_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        # Gate to control how much "thought" updates the state
        # (Similar to LSTM update gate, helps stability)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len, d_model) - Initial Latent (System 1 output)
        Returns:
            x: Refined Latent (System 2 output)
        """
        # Iterative Refinement
        # We unroll the loop for 'thinking_steps'
        
        current_thought = x
        
        for _ in range(self.thinking_steps):
            # Pre-norm
            normed = self.norm(current_thought)
            
            # Compute update candidate
            update = self.think_mlp(normed)
            
            # Compute Gate (0 to 1)
            # Decides how much of the new thought to accept
            g = torch.sigmoid(self.gate(normed))
            
            # Residual Update: z_{t+1} = z_t + gate * update
            current_thought = current_thought + (g * update)
            
        return current_thought
