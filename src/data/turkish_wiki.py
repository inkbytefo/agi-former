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
        Download Turkish Wikipedia using Hugging Face datasets.
        Much more reliable than manual dump parsing.
        """
        print("Downloading Turkish Wikipedia via Hugging Face datasets...")
        
        try:
            from datasets import load_dataset
            
            # Load Turkish Wikipedia (20220301 snapshot - stable)
            print("Loading dataset (this may take a few minutes)...")
            dataset = load_dataset(
                "wikipedia",
                "20220301.tr",
                split="train",
                trust_remote_code=True
            )
            
            print(f"Loaded {len(dataset):,} articles")
            
            # Extract text and convert to bytes
            print("Converting to byte format...")
            all_text = []
            
            # Take first 50k articles for speed (similar scale to enwik8)
            sample_size = min(50000, len(dataset))
            
            for i in range(sample_size):
                if i % 5000 == 0:
                    print(f"  Processed {i}/{sample_size} articles...")
                
                article = dataset[i]
                text = article['text']
                all_text.append(text)
            
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
            
            print("✅ Turkish Wikipedia download complete!")
            
        except ImportError:
            print("ERROR: 'datasets' library not found.")
            print("Install with: pip install datasets")
            raise
        except Exception as e:
            print(f"Error downloading Turkish Wikipedia: {e}")
            raise
    
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

    """
    Turkish Wikipedia Dataset for byte-level language modeling.
    Comparable to enwik8 format for benchmarking.
    """
    def __init__(self, data_dir="./data", split="train", seq_len=1024, download=True):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.seq_len = seq_len
        
        os.makedirs(data_dir, exist_ok=True)
        
        # File paths
        self.raw_file = os.path.join(data_dir, "trwiki_raw.txt")
        
        # Download if needed
        if download and not os.path.exists(self.raw_file):
            self._download_and_process()
        
        # Load data
        if not os.path.exists(self.raw_file):
            raise FileNotFoundError(
                f"Turkish Wikipedia data not found at {self.raw_file}. "
                "Set download=True to download automatically."
            )
        
        with open(self.raw_file, 'rb') as f:
            self.data = f.read()
        
        # Split data (90% train, 5% val, 5% test - same as enwik8)
        total_len = len(self.data)
        train_len = int(0.9 * total_len)
        val_len = int(0.05 * total_len)
        
        if split == "train":
            self.data = self.data[:train_len]
        elif split == "val":
            self.data = self.data[train_len:train_len + val_len]
        elif split == "test":
            self.data = self.data[train_len + val_len:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        print(f"Loaded Turkish Wikipedia ({split}): {len(self.data):,} bytes")
    
    def _download_and_process(self):
        """
        Download Turkish Wikipedia dump and process to plain text.
        Note: This is a simplified version. Full processing requires WikiExtractor.
        """
        print("Downloading Turkish Wikipedia...")
        
        # URL to Turkish Wikipedia dump (latest articles)
        # Using a small subset for demo - full dump is ~3GB compressed
        url = "https://dumps.wikimedia.org/trwiki/latest/trwiki-latest-pages-articles1.xml-p1p187422.bz2"
        
        compressed_file = os.path.join(self.data_dir, "trwiki.xml.bz2")
        
        try:
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, compressed_file)
            print("Download complete.")
            
            # Decompress
            import bz2
            print("Decompressing...")
            with bz2.open(compressed_file, 'rb') as f_in:
                xml_content = f_in.read()
            
            # Extract text from XML
            print("Extracting text...")
            text = self._extract_text_from_xml(xml_content)
            
            # Save as raw bytes
            with open(self.raw_file, 'wb') as f:
                f.write(text.encode('utf-8'))
            
            print(f"Processed {len(text):,} characters to {self.raw_file}")
            
            # Cleanup
            os.remove(compressed_file)
            
        except Exception as e:
            print(f"Error downloading Turkish Wikipedia: {e}")
            print("Please download manually or use a smaller test file.")
            raise
    
    def _extract_text_from_xml(self, xml_content):
        """
        Simple text extraction from Wikipedia XML.
        Removes markup but keeps structure similar to enwik8.
        """
        # Convert bytes to string
        xml_str = xml_content.decode('utf-8', errors='ignore')
        
        # Clean up (basic - not as sophisticated as WikiExtractor)
        # Remove XML tags but keep some structure
        text = re.sub(r'<[^>]+>', '', xml_str)
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return '\n'.join(lines)
    
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
