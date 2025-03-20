import time

from imputegap.wrapper.AlgoPython.HKMF_T.recoveryHKMFT import recoveryHKMFT


def hkmf_t(incomp_data, tags=None, data_names=None, epoch=10, logs=True):
    """
    Perform imputation using Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    tags : numpy.ndarray, optional
        An array containing tags that provide additional structure or metadata about
        the input data. If None, no tags are used (default is None).

    data_names : list of str, optional
        List of names corresponding to each row or column of the dataset for interpretability.
        If None, names are not used (default is None).

    epoch : int, optional
        The maximum number of training epochs for the Hankel Matrix Factorization algorithm.
        If convergence is reached earlier, the process stops (default is 10).

    logs : bool, optional
        Whether to log the execution time (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Example
    -------
    >>> recov_data = hkmf_t(incomp_data, tags=None, data_names=None, epoch=10)
    >>> print(recov_data)

    References
    ----------
    L. Wang, S. Wu, T. Wu, X. Tao and J. Lu, "HKMF-T: Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization," in IEEE Transactions on Knowledge and Data Engineering, vol. 33, no. 11, pp. 3582-3593, 1 Nov. 2021, doi: 10.1109/TKDE.2020.2971190. keywords: {Time series analysis;Matrix decomposition;Market research;Meteorology;Sparse matrices;Indexes;Software;Tagged time series;missing value imputation;blackouts;hankel matrix factorization}
    https://github.com/wangliang-cs/hkmf-t?tab=readme-ov-file
    """
    start_time = time.time()  # Record start time

    recov_data = recoveryHKMFT(miss_data=incomp_data,tags=tags, data_names=data_names, epoch=epoch)

    end_time = time.time()
    if logs:
        print(f"\n\t\t> logs, imputation hkmf_t - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
