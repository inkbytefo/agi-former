import os
import numpy as np

def inspect_data():
    data_dir = "./data"
    file_path = os.path.join(data_dir, "enwik8")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return
        
    print(f"Reading first 200 bytes from {file_path}...")
    
    with open(file_path, 'rb') as f:
        data = f.read(200)
        
    print("-" * 40)
    print(f"Raw Bytes: {[b for b in data]}")
    print("-" * 40)
    
    try:
        text = data.decode('utf-8', errors='replace')
        print(f"Decoded Text:\n{text}")
    except Exception as e:
        print(f"Decoding failed: {e}")
        
    print("-" * 40)

if __name__ == "__main__":
    inspect_data()
