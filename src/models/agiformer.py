## Developer: inkbytefo
## Modified: 2025-11-23

import torch
import torch.nn as nn
from typing import Optional
from .encoder import ByteLatentEncoder
from .layers import HybridBlock
from .reasoning import RecurrentReasoningBlock

class LocalAutoregressiveHead(nn.Module):
    def __init__(self, d_model, patch_size, hidden_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj_latent = nn.Linear(d_model, hidden_dim)
        self.byte_emb = nn.Embedding(256, hidden_dim)
        self.rnn = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 256)

    def forward(self, latents, target_bytes=None, temperature=0.0):
        B, N, D = latents.shape
        latent_context = self.proj_latent(latents).view(B * N, 1, -1)
        
        if target_bytes is not None:
            targets = target_bytes.view(B, N, self.patch_size)
            flat_targets = targets.contiguous().view(B * N, self.patch_size)
            sos = torch.zeros(B * N, 1, dtype=torch.long, device=latents.device)
            rnn_inputs_bytes = torch.cat([sos, flat_targets[:, :-1]], dim=1)
            emb = self.byte_emb(rnn_inputs_bytes)
            latent_expanded = latent_context.expand(-1, self.patch_size, -1)
            rnn_input = torch.cat([emb, latent_expanded], dim=-1)
            out, _ = self.rnn(rnn_input)
            logits = self.head(out)
            return logits.view(B, N, self.patch_size, 256)
        else:
            # Inference logic (omitted for brevity, same as before)
            # ...
            return self._inference(latents, latent_context, temperature)

    def _inference(self, latents, latent_context, temperature):
        # Helper for inference to keep code clean
        B, N, _ = latents.shape
        pred_bytes = []
        current_input = torch.zeros(B * N, 1, dtype=torch.long, device=latents.device)
        hidden = None 
        for i in range(self.patch_size):
            emb = self.byte_emb(current_input)
            rnn_in = torch.cat([emb, latent_context], dim=-1)
            out, hidden = self.rnn(rnn_in, hidden)
            logit = self.head(out)
            if temperature > 0:
                probs = torch.nn.functional.softmax(logit / temperature, dim=-1)
                next_byte = torch.multinomial(probs.squeeze(1), 1)
            else:
                next_byte = torch.argmax(logit, dim=-1)
            pred_bytes.append(next_byte)
            current_input = next_byte
        return torch.cat(pred_bytes, dim=1).view(B, N, self.patch_size)

class AGIFORMER(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        num_heads: int = 8,
        patch_size: int = 4,
        window_size: int = 128,
        vocab_size: int = 256,
        dropout: float = 0.1,
        thinking_steps: int = 3
    ):
        super().__init__()
        
        self.encoder = ByteLatentEncoder(d_model, patch_size, dropout)
        
        # Hybrid Blocks now use Hebbian Memory
        self.layers = nn.ModuleList([
            HybridBlock(d_model, num_heads, window_size, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.reasoning = RecurrentReasoningBlock(d_model, thinking_steps, dropout)
        self.head = LocalAutoregressiveHead(d_model, patch_size)

    def forward(self, x, target_bytes=None, temperature=0.0):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        x = self.reasoning(x)
        logits = self.head(x, target_bytes, temperature=temperature)
        return logits
