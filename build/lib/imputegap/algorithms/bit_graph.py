import time

from imputegap.wrapper.AlgoPython.BiTGraph.main import recoveryBitGRAPH


def bit_graph(incomp_data, node_number=-1, kernel_set=[1], dropout=0.1, subgraph_size=5, node_dim=3, seq_len=1, lr=0.001, epoch=10, seed=42, logs=True, verbose=True):
    """
    Perform imputation using Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    node_number : int, optional
        The number of nodes (time series variables) in the dataset. If not provided,
        it is inferred from `incomp_data`. If -1, set automatically from the len of the values

    kernel_set : list, optional
        Set of kernel sizes used in the model for graph convolution operations (default: [1]).

    dropout : float, optional
        Dropout rate applied during training to prevent overfitting (default: 0.1).

    subgraph_size : int, optional
        The size of each subgraph used in message passing within the graph network (default: 5).

    node_dim : int, optional
        Dimensionality of the node embeddings in the graph convolution layers (default: 3).

    seq_len : int, optional
        Length of the input sequence for temporal modeling (default: 1).

    lr : float, optional
        Learning rate for model optimization (default: 0.001).

    epoch : int, optional
        Number of training epochs (default: 10).

    seed : int, optional
        Random seed for reproducibility (default: 42).

    logs : bool, optional
        Whether to log the execution time (default is True).

    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Example
    -------
        >>> recov_data = bit_graph(incomp_data, tags=None, data_names=None, epoch=10)
        >>> print(recov_data)

    References
    ----------
    X. Chen1, X. Li, T. Wu, B. Liu and Z. Li, BIASED TEMPORAL CONVOLUTION GRAPH NETWORK FOR TIME SERIES FORECASTING WITH MISSING VALUES
    https://github.com/chenxiaodanhit/BiTGraph
    """
    start_time = time.time()  # Record start time

    recov_data = recoveryBitGRAPH(input=incomp_data, node_number=node_number, kernel_set=kernel_set, dropout=dropout, subgraph_size=subgraph_size, node_dim=node_dim, seq_len=seq_len, lr=lr, epoch=epoch, seed=seed, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation bit graph - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
