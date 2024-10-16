import time
from imputegap.wrapper.AlgoPython.IIM.testerIIM import impute_with_algorithm


def iim(contamination, number_neighbor, algo_code, logs=True):
    """
    Perform imputation using the Iterative Imputation Method (IIM) algorithm.

    Parameters
    ----------
    contamination : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    number_neighbor : int
        The number of neighbors to use for the K-Nearest Neighbors (KNN) classifier (default is 10).
    algo_code : str
        The specific action code for the IIM output. This determines the behavior of the algorithm.
    logs : bool, optional
        Whether to log the execution time (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Notes
    -----
    The IIM algorithm works by utilizing K-Nearest Neighbors (KNN) to estimate missing values in time series data.
    Depending on the provided `algo_code`, different versions of the algorithm may be executed.

    The function logs the total execution time if `logs` is set to True.

    Example
    -------
    >>> imputed_data = iim(contamination_matrix, number_neighbor=10, algo_code="iim 2")
    >>> print(imputed_data)

    :author: Quentin Nater
    """
    start_time = time.time()  # Record start time

    imputed_matrix = impute_with_algorithm(algo_code, contamination.copy(), number_neighbor)

    end_time = time.time()
    if logs:
        print(f"\n\t\t> logs, imputation iim - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return imputed_matrix
