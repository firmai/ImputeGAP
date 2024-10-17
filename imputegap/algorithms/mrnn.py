import time
from imputegap.wrapper.AlgoPython.MRNN.testerMRNN import mrnn_recov


def mrnn(contamination, hidden_dim, learning_rate, iterations, sequence_length, logs=True):
    """
    Perform imputation using the Multivariate Recurrent Neural Network (MRNN) algorithm.

    Parameters
    ----------
    contamination : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    hidden_dim : int
        The number of hidden dimensions in the MRNN model.
    learning_rate : float
        The learning rate for the training process.
    iterations : int
        The number of iterations for training the MRNN model.
    sequence_length : int
        The length of sequences used within the MRNN model.
    logs : bool, optional
        Whether to log the execution time (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Notes
    -----
    The MRNN algorithm is a machine learning-based approach for time series imputation, where missing values are recovered using a recurrent neural network structure.

    This function logs the total execution time if `logs` is set to True.

    Example
    -------
    >>> imputed_data = mrnn(contamination_matrix, hidden_dim=64, learning_rate=0.001, iterations=1000, sequence_length=7)
    >>> print(imputed_data)

    References
    ----------
    J. Yoon, W. R. Zame and M. van der Schaar, "Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks," in IEEE Transactions on Biomedical Engineering, vol. 66, no. 5, pp. 1477-1490, May 2019, doi: 10.1109/TBME.2018.2874712. keywords: {Time measurement;Interpolation;Estimation;Medical diagnostic imaging;Correlation;Recurrent neural networks;Biomedical measurement;Missing data;temporal data streams;imputation;recurrent neural nets}
    """
    start_time = time.time()  # Record start time

    imputed_matrix = mrnn_recov(matrix_in=contamination, hidden_dim=hidden_dim, learning_rate=learning_rate,
                                iterations=iterations, seq_length=sequence_length)

    end_time = time.time()
    if logs:
        print(f"\n\t\t> logs, imputation mrnn - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return imputed_matrix
