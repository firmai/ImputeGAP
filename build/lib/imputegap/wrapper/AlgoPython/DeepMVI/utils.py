import numpy as np
import copy
from contextlib import contextmanager

@contextmanager
def null_context():
    yield

def is_blackout(matrix):
    arr = (np.sum(np.isnan(matrix).astype(int),axis=1) == matrix.shape[1])
    return arr.astype(int).sum() > 0


def make_validation(matrix, num_missing=20):
    np.random.seed(0)
    nan_mask = np.isnan(matrix)

    # Create indicator matrix
    padded_mat = np.concatenate([np.zeros((1, nan_mask.shape[1])), nan_mask, np.zeros((1, nan_mask.shape[1]))], axis=0)
    indicator_mat = (padded_mat[1:, :] - padded_mat[:-1, :]).T

    # Detect missing block positions
    pos_start = np.where(indicator_mat == 1)
    pos_end = np.where(indicator_mat == -1)

    # Compute lengths of missing blocks
    lens = (pos_end[1] - pos_start[1])[:, None]
    start_index = pos_start[1][:, None]
    time_series = pos_start[0][:, None]
    test_points = np.concatenate([start_index, time_series, lens], axis=1)

    # Compute block size with error handling
    temp = np.copy(test_points[:, 2])
    if len(temp) > 10:
        block_size = temp[int(len(temp) / 10):-int(len(temp) / 10) - 1].mean()
    else:
        block_size = temp.mean() if len(temp) > 0 else 1  # Default to 1 if empty

    #block_size = 1  # CURRENT DEBUG TO CHANGE - NATERQ

    # Compute kernel size w
    w = int(10 * np.log10(max(block_size, 1)))  # Avoid log of zero or negative values
    w = max(w, 1)  # Ensure minimum kernel size of 1

    # Compute final validation block size
    val_block_size = int(min(block_size, w))

    # Adjust number of missing values based on block size
    num_missing = int(num_missing / val_block_size)

    train_matrix = copy.deepcopy(matrix)
    val_points = []

    # Introduce missing values in the training matrix
    for _ in range(num_missing):
        validation_points = np.random.uniform(0, matrix.shape[0] - val_block_size, (matrix.shape[1])).astype(int)
        for i, x in enumerate(validation_points):
            train_matrix[x:x + val_block_size, i] = np.nan
            val_points.append([x, i, val_block_size])

    return train_matrix, matrix, np.array(val_points), test_points, int(block_size), w

