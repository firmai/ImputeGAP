import time

from imputegap.wrapper.AlgoPython.priSTI.runnerPRISTI import recovPRISTI


def pristi(incomp_data, target_strategy="hybrid", unconditional=True, seed=42, device="cpu", logs=True, verbose=True):
    """
    Perform imputation using the priSTI (Probabilistic Imputation via Sequential Targeted Imputation) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    target_strategy : str, optional
        The strategy to use for targeting missing values. Options include: "hybrid", "random", "historical" (default is "hybrid").
    unconditional : bool, optional
        Whether to use an unconditional imputation model (default is True).
        If False, conditional imputation models are used, depending on available data patterns.
    seed : int, optional
        Random seed for reproducibility (default is 42).
    device : str, optional
        The device to perform computation on, e.g., "cpu" or "cuda" (default is "cpu").
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
        >>> recov_data = priSTI(incomp_data=ts_input, target_strategy="hybrid", unconditional=True, seed=42, device="cpu")
        >>> print(recov_data)

    References
    ----------
    M. Liu, H. Huang, H. Feng, L. Sun, B. Du and Y. Fu, "PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation," 2023 IEEE 39th International Conference on Data Engineering (ICDE), Anaheim, CA, USA, 2023, pp. 1927-1939, doi: 10.1109/ICDE55515.2023.00150.
    https://github.com/LMZZML/PriSTI
    """
    start_time = time.time()  # Record start time

    recov_data = recovPRISTI(data=incomp_data, target_strategy=target_strategy, unconditional=unconditional, seed=seed, device=device, num_workers=1, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n\t> logs, imputation priSTI - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
