## Developer: inkbytefo
## Modified: 2025-11-27

import os
import numpy as np

def load_dataset(data_dir: str):
    path = os.path.join(data_dir, "dataset.npy")
    return np.load(path, mmap_mode="r")

def iter_npy_batches(data: np.ndarray, batch_size: int):
    N = data.shape[0]
    for i in range(0, N, batch_size):
        yield data[i:i+batch_size]

