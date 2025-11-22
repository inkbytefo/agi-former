## Developer: inkbytefo
## Modified: 2025-11-22

import torch
from torch.utils.data import Dataset, DataLoader
import os
import zipfile
import urllib.request
import numpy as np

class Enwik8Dataset(Dataset):
    """
    Dataset for enwik8 (Hutter Prize).
    Downloads and loads the first 100MB of Wikipedia XML dump.
    """
    URL = "http://mattmahoney.net/dc/enwik8.zip"
    FILE_NAME = "enwik8"
    
    def __init__(self, data_dir: str, seq_len: int = 1024, split: str = 'train'):
        self.seq_len = seq_len
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, self.FILE_NAME)
        
        if not os.path.exists(self.file_path):
            self._download_and_extract()
            
        # Load data into memory (100MB is small)
        with open(self.file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            
        # Split: 90MB Train, 5MB Val, 5MB Test
        n = len(data)
        tr_split = int(n * 0.9)
        val_split = int(n * 0.95)
        
        if split == 'train':
            self.data = data[:tr_split]
        elif split == 'val':
            self.data = data[tr_split:val_split]
        else:
            self.data = data[val_split:]
            
        self.data = torch.from_numpy(self.data.copy()).long() # Copy to avoid negative stride issues if any
        
    def _download_and_extract(self):
        print(f"Downloading {self.URL}...")
        zip_path = os.path.join(self.data_dir, "enwik8.zip")
        urllib.request.urlretrieve(self.URL, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
            
    def __len__(self):
        # Number of possible sequences
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        # Random sampling is better for generalization in this context, 
        # but standard Dataset uses index. 
        # We can just return the slice at idx.
        
        # Input: [idx : idx + seq_len]
        # Target: [idx + 1 : idx + seq_len + 1] (Standard next token)
        # But our model expects:
        # Input: [idx : idx + seq_len]
        # Target: [idx + patch_size : idx + seq_len + patch_size] (Next Patch)
        
        # Wait, train.py handles the shifting for Next Patch.
        # So we just return a chunk of length SEQ_LEN + PATCH_SIZE?
        # Or just return SEQ_LEN and let train.py handle it?
        
        # train.py expects (seq, _) and does:
        # x = seq[:, :split_idx]
        # y = seq[:, PATCH_SIZE:]
        
        # So we need to provide a sequence of length SEQ_LEN (which includes the target part).
        # Actually, if we want x to be 1024, we need 1024 + patch_size bytes?
        # train.py: split_idx = seq.size(1) - PATCH_SIZE
        # So if seq is 1024, x is 1020, y is 1020.
        
        # Let's return exactly what's needed.
        chunk = self.data[idx : idx + self.seq_len]
        return chunk, chunk # Dummy target, train.py splits it

def get_enwik8_dataloader(data_dir: str, batch_size: int = 32, seq_len: int = 1024, split: str = 'train'):
    dataset = Enwik8Dataset(data_dir, seq_len, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0)
