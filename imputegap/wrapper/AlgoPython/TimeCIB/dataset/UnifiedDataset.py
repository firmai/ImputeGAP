import numpy as np
import torch

is_artificial = ["physionet"]
is_label = ["hmnist", "physionet", "rotated", "adni"]

import numpy as np
import torch

import numpy as np
import torch

class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, test: bool, args, data):
        """
        Splits the input 2D Tensor into train (70%), validation (10%), and test (20%) sets.

        :param train: Boolean, whether to load the training set.
        :param test: Boolean, whether to load the test set.
        :param args: Arguments object.
        :param data: The input 2D Tensor to be split.
        """
        # Shuffle indices for randomness
        num_samples = data.shape[0]
        indices = torch.randperm(num_samples)  # Torch equivalent of np.random.shuffle()

        # Define split sizes
        train_size = int(0.7 * num_samples)
        valid_size = int(0.1 * num_samples)
        test_size = num_samples - train_size - valid_size  # Remaining samples go to test

        # Assign indices for each split
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]

        # Split the original 2D Tensor
        x_train_full = data[train_indices]
        x_valid_full = data[valid_indices]
        x_test_full = data[test_indices]

        # Create missing data (NaNs) for training, validation, and test
        x_train_miss = x_train_full.clone()  # Fix here
        x_valid_miss = x_valid_full.clone()
        x_test_miss = x_test_full.clone()

        # Create missing masks (1 where missing, 0 where present)
        m_train_miss = ~torch.isnan(x_train_miss)
        m_valid_miss = ~torch.isnan(x_valid_miss)
        m_test_miss = ~torch.isnan(x_test_miss)

        # Assign missing values and masks
        if not test:  # Train or Validation
            if train:  # Training set
                x_full, x_miss, m_miss = x_train_full, x_train_miss, m_train_miss
            else:  # Validation set
                x_full, x_miss, m_miss = x_valid_full, x_valid_miss, m_valid_miss
        else:  # Test set
            x_full, x_miss, m_miss = x_test_full, x_test_miss, m_test_miss

        # Assign data to instance variables
        self.x_full = x_full
        self.x_miss = x_miss
        self.m_miss = m_miss
        self.m_artificial = m_miss  # No artificial missing data

        self.is_label = False  # No labels in dataset
        self.y = None
        self.t = torch.arange(x_full.shape[1])  # Time indices

        # Define the DataLoader with a collate function
        collate = Collate(args.imputed, args.time_length, args.dataset, self.is_label)
        self.loader = torch.utils.data.DataLoader(self, collate_fn=collate, batch_size=args.batch_size, shuffle=False)

    def __len__(self):
        return len(self.x_full)

    def __getitem__(self, idx):
        return self.x_full[idx], self.x_miss[idx], self.m_miss[idx], self.m_artificial[idx], None, self.t


class Collate:
    def __init__(self, is_imputed, time_length, dataset, is_label):
        self.is_imputed = is_imputed
        self.time_length = time_length
        self.dataset = dataset
        self.is_label = is_label
        pass

    def __call__(self, batch):
        """
        Returns a minibatch of images.
        """
        batch_size = len(batch)
        x_full, x_miss, m_miss, m_artificial, y, t = [], [], [], [], [], []
    
        for index in range(batch_size):
            x_full.append(batch[index][0])
            x_miss.append(batch[index][1])
            m_miss.append(batch[index][2])
            m_artificial.append(batch[index][3])
            if self.is_label: y.append(batch[index][4])
            t.append(batch[index][5])

        x_full = torch.tensor(np.array([t.cpu().numpy() for t in x_full]), device="cuda" if torch.cuda.is_available() else "cpu")
        x_miss = torch.tensor(np.array([t.cpu().numpy() for t in x_miss]), device="cuda" if torch.cuda.is_available() else "cpu")
        m_miss = torch.tensor(np.array([t.cpu().numpy() for t in m_miss]), device="cuda" if torch.cuda.is_available() else "cpu")
        m_artificial = torch.tensor(np.array([t.cpu().numpy() for t in m_artificial]), device="cuda" if torch.cuda.is_available() else "cpu")

        if self.is_label:
            y = torch.tensor(np.array([t.cpu().numpy() for t in y]), dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            y = None

        t = torch.tensor(np.array([t.cpu().numpy() for t in t]), device="cuda" if torch.cuda.is_available() else "cpu")

        if self.is_imputed == "forward": x_miss = self.forward_imputation(x_miss, m_miss)
        elif self.is_imputed == "mean": x_miss = self.mean_imputation(x_miss, m_miss)

        return (x_full, x_miss, m_miss, m_artificial, y, t)
    
    def mean_imputation(self, x_miss, m_miss):
        x_mean = torch.sum(x_miss, dim=-2, keepdim=True) / torch.sum((~(m_miss.bool())).float(), dim=-2, keepdim=True)
        x_mean = torch.nan_to_num(x_mean)
        x_mean = torch.tile(x_mean, (1, self.time_length, 1))
        m_miss = m_miss.bool()
        x_imputed = torch.where(~m_miss, x_miss, x_mean)

        return x_imputed

    def forward_imputation(self, x_miss, m_miss):
        m_miss = m_miss.bool()
        x_fwd = x_miss.clone()
        x_bwd = x_miss.clone()
        x_is_observed_fwd = ~m_miss.clone().bool()
        x_is_observed_bwd = ~m_miss.clone().bool()

        if self.dataset == "hmnist":
            for t in range(self.time_length-1):
                x_fwd[:,t+1] = torch.where(~m_miss[:,t+1], x_miss[:,t+1], x_fwd[:,t]).bool().float()
                x_is_observed_fwd[:,t+1] = (x_is_observed_fwd[:,t+1] + x_is_observed_fwd[:,t]).bool()
                x_bwd[:,-2-t] = torch.where(~m_miss[:,-2-t], x_miss[:, -2-t], x_bwd[:, -1-t]).bool().float()
                x_is_observed_bwd[:,-2-t] = (x_is_observed_bwd[:,-2-t] + x_is_observed_bwd[:,-1-t]).bool()
            x_imputed = torch.where(~m_miss, x_miss, torch.where(x_is_observed_fwd, x_fwd, torch.where(x_is_observed_bwd, x_bwd, x_miss)))
        else:
            for t in range(self.time_length-1):
                x_fwd[:,t+1] = torch.where(~m_miss[:,t+1], x_miss[:,t+1], x_fwd[:,t]).float()
                x_is_observed_fwd[:,t+1] = (x_is_observed_fwd[:,t+1] + x_is_observed_fwd[:,t]).bool()
                x_bwd[:,-2-t] = torch.where(~m_miss[:,-2-t], x_miss[:, -2-t], x_bwd[:, -1-t]).float()
                x_is_observed_bwd[:,-2-t] = (x_is_observed_bwd[:,-2-t] + x_is_observed_bwd[:,-1-t]).bool()
            x_imputed = torch.where(~m_miss, x_miss, torch.where(x_is_observed_fwd, x_fwd, torch.where(x_is_observed_bwd, x_bwd, x_miss)))

        return x_imputed