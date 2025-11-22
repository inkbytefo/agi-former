import torch
import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.agiformer import AGIFORMER
from src.models.encoder import ByteLatentEncoder
from src.models.layers import HybridBlock

def test_encoder_shape():
    B, L, D = 2, 128, 64
    patch_size = 4
    encoder = ByteLatentEncoder(d_model=D, patch_size=patch_size)
    
    # Input: Random bytes
    x = torch.randint(0, 256, (B, L))
    out = encoder(x)
    
    # Expected output shape: (B, L // patch_size, D)
    expected_shape = (B, L // patch_size, D)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

def test_hybrid_block_shape():
    B, L, D = 2, 32, 64
    block = HybridBlock(d_model=D, num_heads=4, window_size=16)
    
    x = torch.randn(B, L, D)
    out = block(x)
    
    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"

def test_agiformer_forward():
    B, L = 2, 128
    D = 64
    patch_size = 4
    model = AGIFORMER(d_model=D, n_layers=2, patch_size=patch_size)
    
    x = torch.randint(0, 256, (B, L))
    logits = model(x)
    
    # Expected output: (B, N_Patches, Patch_Size, 256)
    # N_Patches = L // patch_size
    expected_shape = (B, L // patch_size, patch_size, 256)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

if __name__ == "__main__":
    test_encoder_shape()
    test_hybrid_block_shape()
    test_agiformer_forward()
    print("All tests passed!")
