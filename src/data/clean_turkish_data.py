## Developer: inkbytefo
## Modified: 2025-11-22

import os
import re
import torch
from datasets import load_dataset
from tqdm import tqdm

def clean_wiki_text(text):
    """
    Wikipedia metinleri için özel temizlik.
    Dipnotları, parantez içi referansları ve 'Dosya:' gibi meta verileri temizler.
    """
    # 1. [1], [kaynak belirtilmeli] gibi referansları sil
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # 2. (İngilizce: ...), (d. 1990) gibi parantezleri koru ama içindeki garip kodları temizle
    # Basit html tag temizliği
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Gereksiz boşlukları ve satır sonlarını düzelt
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_clean_turkish_data(data_dir="./data", target_mb=150):
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "trwiki_clean_train.bin")
    val_path = os.path.join(data_dir, "trwiki_clean_val.bin")
    
    if os.path.exists(output_path):
        print(f"Clean data already exists at {output_path}")
        return

    print(f"Downloading OFFICIAL Wikipedia (Turkish) dataset...")
    # "20220301.tr" config'i standarttır.
    try:
        dataset = load_dataset("wikipedia", "20220301.tr", split="train", streaming=True, trust_remote_code=True)
    except:
        print("Fallback: Using 'wikimedia/wikipedia' dataset...")
        dataset = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    
    collected_bytes = []
    total_bytes = 0
    target_size = target_mb * 1024 * 1024
    
    print("Processing Wikipedia articles (High Quality)...")
    pbar = tqdm(total=target_mb, unit="MB")
    
    for i, article in enumerate(dataset):
        raw_text = article['text']
        
        # Çok kısa makaleleri (taslakları) atla
        if len(raw_text) < 1000:
            continue
            
        cleaned = clean_wiki_text(raw_text)
        
        # Encode
        encoded = cleaned.encode('utf-8')
        
        # Makaleleri ayırmak için özel ayırıcı (Byte seviyesinde)
        # \n\n (Yeni paragraf) yeterlidir.
        collected_bytes.append(encoded)
        collected_bytes.append(b'\n\n') 
        
        chunk_size = len(encoded) + 2
        total_bytes += chunk_size
        pbar.update(chunk_size / (1024 * 1024))
        
        if total_bytes >= target_size:
            break
            
    pbar.close()
    
    # Flatten
    print("Saving binary files...")
    full_data = b"".join(collected_bytes)
    
    # Split 95/5
    split_idx = int(len(full_data) * 0.95)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    with open(output_path, "wb") as f:
        f.write(train_data)
    with open(val_path, "wb") as f:
        f.write(val_data)
        
    print(f"✅ Dataset Ready: {len(train_data)/1e6:.1f}MB Train, {len(val_data)/1e6:.1f}MB Val")

# Dataset Sınıfı (Aynı kalabilir, sadece dosya adlarını doğru kullanmalı)
class CleanTurkishDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len=1024):
        with open(data_path, "rb") as f:
            self.data = f.read()
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 4)
        
    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 4]
        x = torch.tensor(list(chunk[:-4]), dtype=torch.long)
        y = torch.tensor(list(chunk[4:]), dtype=torch.long)
        return x, y

def get_clean_loader(data_dir, batch_size, seq_len, split="train"):
    path = os.path.join(data_dir, f"trwiki_clean_{split}.bin")
    if not os.path.exists(path):
        # Auto-prepare if missing
        prepare_clean_turkish_data(data_dir)
        
    dataset = CleanTurkishDataset(path, seq_len)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split=="train"), 
        num_workers=0,
        pin_memory=True
    )

if __name__ == "__main__":
    # Gerekli kütüphaneyi yükle
    os.system("pip install datasets apache_beam mwparserfromhell")
    prepare_clean_turkish_data()
