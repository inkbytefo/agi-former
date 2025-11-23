## Developer: inkbytefo
## Modified: 2025-11-23

import torch
import torch.nn.functional as F
from src.models.agiformer import AGIFORMER
import os
import numpy as np

def inspect_system_2(model_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Config (Scaled v2.0)
    D_MODEL = 768
    N_LAYERS = 12
    NUM_HEADS = 12
    PATCH_SIZE = 4
    WINDOW_SIZE = 256
    THINKING_STEPS = 3
    
    print(f"Inspecting {model_path} on {DEVICE}...")
    
    model = AGIFORMER(
        d_model=D_MODEL, 
        n_layers=N_LAYERS, 
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        window_size=WINDOW_SIZE,
        thinking_steps=THINKING_STEPS
    ).to(DEVICE)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print(f"Model {model_path} not found! Using random weights for demo.")
        model.eval()
    
    # Hook mekanizması: Reasoning bloğundaki gate ve update değerlerini yakalayalım
    stats = {"z_diff": []}
    
    def hook_fn(module, input, output):
        # Input is tuple (x,), output is refined x
        z_in = input[0]
        z_out = output
        
        # Measure how much the latent vector changed
        # L2 Distance per token
        diff = torch.norm(z_out - z_in, dim=-1).mean().item()
        stats["z_diff"].append(diff)
    
    # Register hook on the reasoning block
    handle = model.reasoning.register_forward_hook(hook_fn)
    
    # Dummy Input (from enwik8 context)
    dummy_text = "The history of artificial intelligence"
    input_bytes = [ord(c) for c in dummy_text]
    # Pad
    pad = (4 - len(input_bytes) % 4) % 4
    input_bytes.extend([32]*pad)
    
    x = torch.tensor(input_bytes, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Run forward pass triggers the hook
        _ = model(x)
        
    # Manual Inspection of Internal Reasoning Weights
    gate_bias_mean = model.reasoning.gate.bias.mean().item()
    halt_bias_mean = model.reasoning.halt_unit.bias.mean().item()
    
    print("\n--- SYSTEM 2 DIAGNOSTICS (v2.0) ---")
    print(f"1. Latent Refinement (Thinking Magnitude):")
    print(f"   Average Euclidean Distance (z_out - z_in): {np.mean(stats['z_diff']):.4f}")
    print(f"   (If close to 0.0, the model is SKIPPING the thinking step.)")
    
    print(f"\n2. Gate Statistics:")
    print(f"   Update Gate Bias Mean: {gate_bias_mean:.4f}")
    print(f"   Halt Unit Bias Mean:   {halt_bias_mean:.4f}")
    print(f"   (Positive Halt Bias -> Prefers to STOP thinking)")
    print(f"   (Negative Halt Bias -> Prefers to CONTINUE thinking)")
    
    print(f"\n3. Parameter Health:")
    mlp_weight_grad = model.reasoning.think_mlp[0].weight.std().item()
    print(f"   MLP Weight Std: {mlp_weight_grad:.4f}")
    
    # Interpretation
    avg_diff = np.mean(stats['z_diff'])
    if avg_diff < 0.01:
        print("\n[RESULT] SYSTEM 2 IS DORMANT (Collapsed).")
        print("Reason: The model learned that 'not thinking' is safer for loss.")
    elif avg_diff > 20.0:
        print("\n[RESULT] SYSTEM 2 IS UNSTABLE (Exploding).")
    else:
        print("\n[RESULT] SYSTEM 2 IS ACTIVE.")
        print("The model is actively modifying its latent state.")
    
    # Cleanup
    handle.remove()

if __name__ == "__main__":
    inspect_system_2("best_model_scaled.pth")
