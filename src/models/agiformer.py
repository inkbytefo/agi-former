## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
from typing import Optional
from .encoder import ByteLatentEncoder
from .layers import HybridBlock

class LocalAutoregressiveHead(nn.Module):
    """
    Latent vector -> Bytes (Autoregressive).
    Global Model -> Latent -> Local Model -> Bytes
    """
    def __init__(self, d_model, patch_size, hidden_dim=256):
        super().__init__()
        self.patch_size = patch_size
        
        # Project latent to be the initial state or context
        self.proj_latent = nn.Linear(d_model, hidden_dim)
        
        # Byte embedding for the local decoder
        self.byte_emb = nn.Embedding(256, hidden_dim)
        
        # Small, fast RNN (GRU) for local decoding
        # Input size is now hidden_dim (embedding) + hidden_dim (latent context)
        self.rnn = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        
        self.head = nn.Linear(hidden_dim, 256)

    def forward(self, latents, target_bytes=None):
        """
        Args:
            latents: (B, N_Patches, D_Model)
            target_bytes: (B, L) - Required for training (Teacher Forcing)
        """
        B, N, D = latents.shape
        
        # (B * N, 1, Hidden)
        latent_context = self.proj_latent(latents).view(B * N, 1, -1)
        
        if target_bytes is not None:
            # TRAINING MODE (Teacher Forcing)
            # Reshape targets to (B, N, Patch_Size)
            targets = target_bytes.view(B, N, self.patch_size)
            
            # Flatten: (B*N, Patch_Size)
            flat_targets = targets.contiguous().view(B * N, self.patch_size)
            
            # Shift targets right to get inputs
            sos = torch.zeros(B * N, 1, dtype=torch.long, device=latents.device)
            rnn_inputs_bytes = torch.cat([sos, flat_targets[:, :-1]], dim=1) # (B*N, P)
            
            emb = self.byte_emb(rnn_inputs_bytes) # (B*N, P, Hidden)
            
            # Concatenate latent context to every step
            latent_expanded = latent_context.expand(-1, self.patch_size, -1)
            
            # Concatenation instead of addition to preserve signal
            rnn_input = torch.cat([emb, latent_expanded], dim=-1) # (B*N, P, Hidden * 2)
            
            out, _ = self.rnn(rnn_input)
            logits = self.head(out) # (B*N, P, 256)
            
            return logits.view(B, N, self.patch_size, 256)
            
        else:
            # INFERENCE MODE
            pred_bytes = []
            # Start with SOS (0)
            current_input = torch.zeros(B * N, 1, dtype=torch.long, device=latents.device)
            
            # Initialize hidden state
            hidden = None # Let GRU initialize to 0 or we could use latent as initial state if mapped correctly
            
            for i in range(self.patch_size):
                emb = self.byte_emb(current_input) # (B*N, 1, H)
                
                # Concatenate latent
                rnn_in = torch.cat([emb, latent_context], dim=-1) # (B*N, 1, H*2)
                
                out, hidden = self.rnn(rnn_in, hidden)
                logit = self.head(out) # (B*N, 1, 256)
                
                # Greedy decode
                next_byte = torch.argmax(logit, dim=-1)
                pred_bytes.append(next_byte)
                current_input = next_byte
            
            return torch.cat(pred_bytes, dim=1).view(B, N, self.patch_size)

class AGIFORMER(nn.Module):
    """
    AGIFORMER: A Byte-Latent Hybrid Architecture.
    """
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        num_heads: int = 8,
        patch_size: int = 4,
        window_size: int = 128,
        vocab_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = ByteLatentEncoder(
            d_model=d_model,
            patch_size=patch_size,
            dropout=dropout
        )
        
        self.layers = nn.ModuleList([
            HybridBlock(
                d_model=d_model,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        
        # Local Autoregressive Head
        self.head = LocalAutoregressiveHead(d_model, patch_size)

    def forward(self, x: torch.Tensor, target_bytes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq_Len) uint8 - Input Context
            target_bytes: (Batch, Seq_Len_Target) - Required for training the local head
            
        Returns:
            logits: (Batch, Num_Patches, Patch_Size, 256)
        """
        # 1. Encode
        x = self.encoder(x) # (B, N_Patches, D)
        
        # 2. Backbone
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        
        # 3. Head (Local Autoregressive)
        logits = self.head(x, target_bytes)
        
        return logits
