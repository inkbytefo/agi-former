import gradio as gr
import torch
import torch.nn.functional as F
from src.models.agiformer import AGIFORMER
import os

# Load Model (v2.0 Scaled Configuration)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AGIFORMER(
    d_model=768, 
    n_layers=12, 
    num_heads=12, 
    patch_size=4, 
    window_size=256, 
    thinking_steps=3
).to(DEVICE)

MODEL_PATH = "best_model_scaled.pth"
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
else:
    print("Warning: Checkpoint not found. Using random weights.")

def sample(logits, temperature):
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

def generate(prompt, temp, max_new_tokens=200):
    input_bytes = list(prompt.encode('utf-8'))
    pad = (4 - len(input_bytes) % 4) % 4
    input_bytes.extend([32]*pad)
    
    generated = input_bytes[:]
    
    with torch.no_grad():
        # Generate patch by patch (4 bytes at a time)
        for _ in range(max_new_tokens // 4):
            ctx = generated[-1024:] # Sliding window context
            x = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            # v2.0 returns logits: (B, N, 4, 256)
            logits = model(x)
            
            # Get last patch logits: (4, 256)
            last_patch_logits = logits[0, -1, :, :]
            
            # Sample
            new_bytes = sample(last_patch_logits, temp).tolist()
            generated.extend(new_bytes)
            
            # Stream output
            curr_text = bytes(generated).decode('utf-8', errors='replace')
            yield curr_text

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Başlangıç Metni", placeholder="Bir zamanlar..."), 
        gr.Slider(0.0, 1.5, value=0.7, label="Yaratıcılık (Temperature)"),
        gr.Slider(50, 500, value=200, step=10, label="Maksimum Uzunluk")
    ],
    outputs=gr.Textbox(label="AGIFORMER v2.0"),
    title="AGIFORMER v2.0: Byte-Level Turkish AI",
    description="Token kullanmayan, düşünen yapay zeka. (v2.0 Scaled Architecture)",
    examples=[
        ["Türkiye Cumhuriyeti ", 0.7, 200],
        ["Yapay zeka nedir? ", 0.5, 150],
        ["İstanbul'un tarihi ", 0.8, 300]
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
