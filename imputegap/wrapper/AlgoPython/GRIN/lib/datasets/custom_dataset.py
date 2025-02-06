import numpy as np


class CustomDataset:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.training_mask = np.ones_like(self.data, dtype=bool)  # Create a dummy mask
        self.eval_mask = np.ones_like(self.data, dtype=bool)  # Create a dummy mask

    def numpy(self, return_idx=False):
        if return_idx:
            return self.data, np.arange(len(self.data))
        return self.data

    def splitter(self, torch_dataset, val_len=0.1, test_len=0.2):
        num_samples = len(torch_dataset)
        val_size = int(num_samples * val_len)
        test_size = int(num_samples * test_len)
        train_size = num_samples - val_size - test_size

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        return indices[:train_size], indices[train_size:train_size + val_size], indices[train_size + val_size:]

    def get_similarity(self, thr=0.1):
        return np.random.rand(self.data.shape[0], self.data.shape[0]) > thr  # Dummy similarity matrix
