import time

from imputegap.wrapper.AlgoPython.GRIN.scripts.recoveryGRIN import recoveryGRIN
from imputegap.wrapper.AlgoPython.MRNN.runnerMRNN import mrnn_recov


def grin(incomp_data, logs=True):
    """
    Perform imputation using the Multivariate Recurrent Neural Network (MRNN) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

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
    >>> recov_data = mrnn(incomp_data, hidden_dim=64, learning_rate=0.001, iterations=1000, sequence_length=7)
    >>> print(recov_data)

    References
    ----------
    A. Cini, I. Marisca, and C. Alippi, "Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks," International Conference on Learning Representations (ICLR), 2022.
    """
    start_time = time.time()  # Record start time

    recov_data = recoveryGRIN(input_data=incomp_data)

    end_time = time.time()
    if logs:
        print(f"\n\t\t> logs, imputation grin - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
