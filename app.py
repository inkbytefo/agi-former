## Developer: inkbytefo
## Modified: 2025-11-27

import gradio as gr
import jax
import os
from jax import random
from src.models.agiformer import agiformer_init
from src.data.morphology import build_vocab
from src.inference import generate_words, invert_vocab

print("⏳ Sözlük oluşturuluyor...")
sample_path = os.path.join("data", "sample.txt")
if not os.path.exists(sample_path):
    texts = ["Merhaba dünya", "Yapay zeka", "Türkiye Cumhuriyeti"]
else:
    with open(sample_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

root2id, suffix2id = build_vocab(texts, root_limit=10000, suffix_limit=500)
id2root = invert_vocab(root2id)
id2suffix = invert_vocab(suffix2id)
SUFFIX_SLOTS = 5

print(f"✅ Sözlük Hazır: {len(root2id)} kök, {len(suffix2id)} ek.")

print("🧠 Model başlatılıyor...")
key = random.PRNGKey(42)
model_params = agiformer_init(
    d_model=128,
    n_layers=4,
    num_heads=4,
    patch_size=4,
    window_size=64,
    thinking_steps=3,
    key=key,
)

def generate_response(prompt, effort, temp_root, temp_suffix):
    try:
        output = generate_words(
            params=model_params,
            prompt_text=prompt,
            root2id=root2id,
            suffix2id=suffix2id,
            id2root=id2root,
            id2suffix=id2suffix,
            suffix_slots=SUFFIX_SLOTS,
            num_words=15,
            effort=effort,
            temperature_root=temp_root,
            temperature_suffix=temp_suffix,
            seed=random.randint(key, (1,), 0, 10000).item(),
        )
        return output
    except Exception as e:
        return f"Hata: {str(e)}"

demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Başlangıç (Prompt)", value="Türkiye"),
        gr.Slider(0.1, 1.0, value=0.6, label="Düşünme Eforu (Effort)"),
        gr.Slider(0.1, 2.0, value=0.8, label="Kök Çeşitliliği (Temp Root)"),
        gr.Slider(0.1, 2.0, value=0.5, label="Ek Tutarlılığı (Temp Suffix)"),
    ],
    outputs=gr.Textbox(label="Morpho-AGI Çıktısı"),
    title="AGIFORMER v3.0: Morpho-Semantic Turkish AI",
    description="Kök ve Ekleri ayrı ayrı işleyen, Türkçe morfolojisine özelleşmiş AGI mimarisi.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
