## Developer: inkbytefo
## Modified: 2025-11-22

import torch
from torch.utils.data import IterableDataset, DataLoader
import random

class SyntheticByteDataset(IterableDataset):
    """
    Generates infinite synthetic byte sequences for testing.
    Task: Copy task or simple pattern repetition to verify memory.
    """
    def __init__(self, seq_len: int = 1024, mode: str = 'repeat'):
        self.seq_len = seq_len
        self.mode = mode

    def __iter__(self):
        while True:
            if self.mode == 'repeat':
                # Generate a random pattern of length 16 and repeat it
                pattern_len = 16
                pattern = torch.randint(0, 256, (pattern_len,), dtype=torch.uint8)
                # Repeat to fill seq_len
                repeats = (self.seq_len // pattern_len) + 1
                seq = pattern.repeat(repeats)[:self.seq_len]
                yield seq.long(), seq.long() # Input, Target (Same for auto-regression)
            
            elif self.mode == 'random':
                # Completely random bytes
                seq = torch.randint(0, 256, (self.seq_len,), dtype=torch.long)
                yield seq, seq

def get_dataloader(batch_size: int = 32, seq_len: int = 1024):
    dataset = SyntheticByteDataset(seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size)
