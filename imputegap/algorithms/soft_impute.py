import time
import ctypes as __native_c_types_import;

from imputegap.tools import utils

def native_soft_impute(__py_matrix, __py_max_rank):
    """
    Perform matrix imputation using the Soft Impute algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).
    __py_max_rank : int
        The max rank for matrix decomposition (must be greater than 0 and less than the number of columns).

    Returns
    -------
    numpy.ndarray
        A completed matrix with missing values imputed using the Soft Impute algorithm.

    References
    ----------
    R. Mazumder, T. Hastie, and R. Tibshirani. Spectral regularization algorithms for learning large incomplete matrices. Journal of Machine Learning Research, 11:2287–2322, 2010.
    """

    shared_lib = utils.load_share_lib("lib_soft_impute.so")

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_max_rank >= 0);
    assert (__py_max_rank < __py_m);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __ctype_rank = __native_c_types_import.c_ulonglong(__py_max_rank);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.recoverySoftImpute(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_rank);

    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def soft_impute(incomp_data, max_rank, logs=True, lib_path=None):
    """
    Soft Impute algorithm for matrix imputation.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    max_rank : int
        The max rank for matrix decomposition (must be greater than 1 and smaller than the number of series).
    logs : bool, optional
        Whether to log the execution time (default is True).
    lib_path : str, optional
        Custom path to the shared library file (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Example
    -------
    >>> recov_data = soft_impute(incomp_data=incomp_data, max_rank=5, logs=True)
    >>> print(recov_data)

    """
    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    recov_data = native_soft_impute(incomp_data, max_rank)

    end_time = time.time()

    if logs:
        print(f"\n\t\t> logs, imputation Soft Impute - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
