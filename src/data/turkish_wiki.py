## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.utils.data as data
import os

class TurkishWikiDataset(data.Dataset):
    """
    Turkish Wikipedia Dataset via Hugging Face datasets.
    Comparable to enwik8 format for benchmarking.
    """
    def __init__(self, data_dir="./data", split="train", seq_len=1024, download=True):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.seq_len = seq_len
        
        os.makedirs(data_dir, exist_ok=True)
        
        # File paths
        self.processed_file = os.path.join(data_dir, f"trwiki_{split}.bin")
        
        # Download if needed
        if download and not os.path.exists(self.processed_file):
            self._download_and_process()
        
        # Load data
        if not os.path.exists(self.processed_file):
            raise FileNotFoundError(
                f"Turkish Wikipedia data not found at {self.processed_file}. "
                "Set download=True to download automatically."
            )
        
        with open(self.processed_file, 'rb') as f:
            self.data = f.read()
        
        print(f"Loaded Turkish Wikipedia ({split}): {len(self.data):,} bytes")
    
    def _download_and_process(self):
        """
        Download Turkish text using allenai/c4 (Parquet format).
        Modern, maintained, no loading scripts required.
        """
        print("Downloading Turkish text via allenai/c4...")
        
        try:
            from datasets import load_dataset
            
            # Load allenai/c4 Turkish subset (Parquet - no scripts)
            print("Loading allenai/c4 Turkish corpus (streaming)...")
            dataset = load_dataset(
                "allenai/c4",
                "tr",  # Turkish language code
                split="train",
                streaming=True
            )
            
            print("Converting to byte format...")
            all_text = []
            
            # Take enough text to match enwik8 scale (~100MB)
            target_bytes = 100_000_000
            current_bytes = 0
            count = 0
            
            for example in dataset:
                text = example['text']
                
                # Clean: remove empty or very short texts
                if len(text.strip()) < 50:
                    continue
                
                all_text.append(text)
                current_bytes += len(text.encode('utf-8'))
                count += 1
                
                if count % 1000 == 0:
                    mb = current_bytes / 1e6
                    print(f"  Processed {count} texts ({mb:.1f} MB)...")
                
                if current_bytes >= target_bytes:
                    break
            
            print(f"Collected {count} texts")
            
            # Join all text
            full_text = '\n\n'.join(all_text)
            
            # Convert to bytes (UTF-8)
            text_bytes = full_text.encode('utf-8')
            
            print(f"Total: {len(text_bytes):,} bytes ({len(text_bytes) / 1e6:.1f} MB)")
            
            # Split: 90% train, 5% val, 5% test (same as enwik8)
            total_len = len(text_bytes)
            train_len = int(0.9 * total_len)
            val_len = int(0.05 * total_len)
            
            splits = {
                'train': text_bytes[:train_len],
                'val': text_bytes[train_len:train_len + val_len],
                'test': text_bytes[train_len + val_len:]
            }
            
            # Save each split
            for split_name, split_bytes in splits.items():
                filepath = os.path.join(self.data_dir, f"trwiki_{split_name}.bin")
                with open(filepath, 'wb') as f:
                    f.write(split_bytes)
                print(f"Saved {split_name}: {len(split_bytes):,} bytes")
            
            print("✅ Turkish text download complete!")
            
        except ImportError:
            print("ERROR: 'datasets' library not found.")
            print("Install with: pip install datasets")
            raise
        except Exception as e:
            print(f"Error downloading Turkish text: {e}")
            print("\nFallback: Creating small test dataset...")
            self._create_test_dataset()
    
    def _create_test_dataset(self):
        """
        Create a small Turkish test dataset from hardcoded text.
        For testing when download fails.
        """
        turkish_sample = """
Türkiye, Avrupa ve Asya kıtalarında yer alan bir ülkedir. Başkenti Ankara'dır. 
En kalabalık şehri İstanbul'dur. Türkiye'nin tarihi çok eskidir. Anadolu, tarih 
boyunca birçok medeniyete ev sahipliği yapmıştır. Hitit, Frig, Lidya, Pers, 
Roma, Bizans ve Osmanlı gibi imparatorluklar bu topraklarda hüküm sürmüştür.

Türk dili, Altay dil ailesinin Türk koluna aittir. Sondan eklemeli bir dildir. 
Bu özellik, İngilizce gibi analitik dillerden farklı olarak, kelimelere ekler 
eklenerek anlam zenginleştirilmesine olanak tanır. Örneğin: kitap, kitaplar, 
kitaplarım, kitaplarımdan gibi çeşitli formlar oluşturulabilir.

Türkiye'nin coğrafyası çok çeşitlidir. Doğu Anadolu'da yüksek dağlar ve platolar 
bulunurken, Ege ve Akdeniz kıyılarında ılıman iklim hakimdir. Karadeniz bölgesi 
yağışlı ve yeşildir. Güneydoğu Anadolu ise daha kurak bir bölgedir.
""" * 1000  # Repeat to get more data
        
        text_bytes = turkish_sample.encode('utf-8')
        
        # Create minimal splits
        total_len = len(text_bytes)
        train_len = int(0.9 * total_len)
        val_len = int(0.05 * total_len)
        
        splits = {
            'train': text_bytes[:train_len],
            'val': text_bytes[train_len:train_len + val_len],
            'test': text_bytes[train_len + val_len:]
        }
        
        for split_name, split_bytes in splits.items():
            filepath = os.path.join(self.data_dir, f"trwiki_{split_name}.bin")
            with open(filepath, 'wb') as f:
                f.write(split_bytes)
            print(f"Created test {split_name}: {len(split_bytes):,} bytes")
        
        print("⚠️ Using test dataset (limited Turkish text)")
    
    def __len__(self):
        # Number of possible sequences
        return max(0, len(self.data) - 2 * self.seq_len)
    
    def __getitem__(self, idx):
        """
        Returns:
            input: (seq_len,) - Context bytes
            target: (seq_len,) - Target bytes (next patch)
        """
        # Input context
        start_idx = idx
        end_idx = start_idx + self.seq_len
        
        # Target is shifted by patch_size (4 bytes default)
        target_start = start_idx + 4
        target_end = target_start + self.seq_len
        
        # Extract bytes
        input_bytes = torch.tensor(
            list(self.data[start_idx:end_idx]),
            dtype=torch.long
        )
        
        target_bytes = torch.tensor(
            list(self.data[target_start:target_end]),
            dtype=torch.long
        )
        
        return input_bytes, target_bytes


def get_turkish_wiki_dataloader(batch_size, seq_len, split="train"):
    """
    Create DataLoader for Turkish Wikipedia.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        split: "train", "val", or "test"
    
    Returns:
        DataLoader yielding (input, target) batches
    """
    dataset = TurkishWikiDataset(
        data_dir="./data",
        split=split,
        seq_len=seq_len,
        download=True
    )
    
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True
    )
    
    return loader

