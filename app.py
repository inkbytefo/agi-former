import gradio as gr
import torch
from src.models.agiformer import AGIFORMER
import os

# Load Model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AGIFORMER(d_model=512, n_layers=6, patch_size=4, thinking_steps=3).to(DEVICE)

if os.path.exists("best_model_turkish.pth"):
    model.load_state_dict(torch.load("best_model_turkish.pth", map_location=DEVICE))
    model.eval()

def generate(prompt, temp):
    # ... (generate.py içindeki mantığın aynısı buraya) ...
    # Basitlik için kısa versiyon:
    input_bytes = list(prompt.encode('utf-8'))
    pad = (4 - len(input_bytes) % 4) % 4
    input_bytes.extend([32]*pad)
    
    generated = input_bytes[:]
    
    with torch.no_grad():
        for _ in range(50): # 200 bytes approx
            ctx = generated[-1024:]
            x = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(DEVICE)
            pred = model(x, temperature=temp)
            generated.extend(pred[0, -1, :].cpu().tolist())
            
            # Stream output
            curr_text = bytes(generated).decode('utf-8', errors='replace')
            yield curr_text

demo = gr.Interface(
    fn=generate,
    inputs=[gr.Textbox(label="Başlangıç Metni"), gr.Slider(0.1, 1.5, value=0.7, label="Yaratıcılık (Temperature)")],
    outputs=gr.Textbox(label="AGIFORMER"),
    title="AGIFORMER: Byte-Level Turkish AI",
    description="Token kullanmayan, düşünen yapay zeka."
)

if __name__ == "__main__":
    demo.launch()
