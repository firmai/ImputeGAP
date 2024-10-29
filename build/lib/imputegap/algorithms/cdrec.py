import ctypes
import time
import ctypes as __native_c_types_import;
import numpy as __numpy_import;

from imputegap.tools import utils


def __marshal_as_numpy_column(__ctype_container, __py_sizen, __py_sizem):
    """
    Marshal a ctypes container as a numpy column-major array.

    Parameters
    ----------
    __ctype_container : ctypes.Array
        The input ctypes container (flattened matrix).
    __py_sizen : int
        The number of rows in the numpy array.
    __py_sizem : int
        The number of columns in the numpy array.

    Returns
    -------
    numpy.ndarray
        A numpy array reshaped to the original matrix dimensions (row-major order).
    """
    __numpy_marshal = __numpy_import.array(__ctype_container).reshape(__py_sizem, __py_sizen).T;

    return __numpy_marshal;


def __marshal_as_native_column(__py_matrix):
    """
    Marshal a numpy array as a ctypes flat container for passing to native code.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input numpy matrix (2D array).

    Returns
    -------
    ctypes.Array
        A ctypes array containing the flattened matrix (in column-major order).
    """
    __py_input_flat = __numpy_import.ndarray.flatten(__py_matrix.T);
    __ctype_marshal = __numpy_import.ctypeslib.as_ctypes(__py_input_flat);

    return __ctype_marshal;


def native_cdrec(__py_matrix, __py_rank, __py_epsilon, __py_iterations):
    """
    Perform matrix imputation using the CDRec algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).
    __py_rank : int
        The truncation rank for matrix decomposition (must be greater than 0 and less than the number of columns).
    __py_epsilon : float
        The epsilon value, used as the threshold for stopping iterations based on difference.
    __py_iterations : int
        The maximum number of allowed iterations for the algorithm.

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation.

    References
    ----------
    Khayati, M., Cudré-Mauroux, P. & Böhlen, M.H. Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl Inf Syst 62, 2257–2280 (2020). https://doi.org/10.1007/s10115-019-01421-7
    """

    shared_lib = utils.load_share_lib("lib_cdrec.so")

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_rank >= 0);
    assert (__py_rank < __py_m);
    assert (__py_epsilon > 0);
    assert (__py_iterations > 0);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __ctype_rank = __native_c_types_import.c_ulonglong(__py_rank);
    __ctype_epsilon = __native_c_types_import.c_double(__py_epsilon);
    __ctype_iterations = __native_c_types_import.c_ulonglong(__py_iterations);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = __marshal_as_native_column(__py_matrix);

    shared_lib.cdrec_imputation_parametrized(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_rank, __ctype_epsilon, __ctype_iterations);

    __py_imputed_matrix = __marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def cdrec(contamination, truncation_rank, iterations, epsilon, logs=True, lib_path=None):
    """
    CDRec algorithm for matrix imputation of missing values using Centroid Decomposition.

    Parameters
    ----------
    contamination : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    truncation_rank : int
        The truncation rank for matrix decomposition (must be greater than 1 and smaller than the number of series).
    epsilon : float
        The learning rate (stopping criterion threshold).
    iterations : int
        The maximum number of iterations allowed for the algorithm.
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
    >>> imputed_data = cdrec(contamination=contamination_matrix, truncation_rank=1, iterations=100, epsilon=0.000001, logs=True)
    >>> print(imputed_data)

    """
    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    imputed_matrix = native_cdrec(contamination, truncation_rank, epsilon, iterations)

    end_time = time.time()

    if logs:
        print(f"\n\t\t> logs, imputation cdrec - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return imputed_matrix
