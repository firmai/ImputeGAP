import ctypes
import time
import os
import importlib.resources
import ctypes as __native_c_types_import;
import numpy as __numpy_import;

def __marshal_as_numpy_column(__ctype_container, __py_sizen, __py_sizem):
    """
    Convert a ctypes container to a numpy array in column-major order.

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
    Convert a numpy array to a ctypes flat container for passing to native code.

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


def load_share_lib(name = "lib_stmvl", lib=True):
    """
    Load the shared library based on the operating system.

    Parameters
    ----------
    name : str, optional
        The name of the shared library (default is "lib_stmvl").
    lib : bool, optional
        If True, the function loads the library from the default 'imputegap' path; if False, it loads from a local path (default is True).

    Returns
    -------
    ctypes.CDLL
        The loaded shared library object.
    """

    if lib:
        lib_path = importlib.resources.files('imputegap.algorithms.lib').joinpath("./lib_stmvl.so")
    else:
        local_path_lin = './algorithms/lib/'+name+'.so'

        if not os.path.exists(local_path_lin):
            local_path_lin = './imputegap/algorithms/lib/'+name+'.so'

        lib_path = os.path.join(local_path_lin)

    return ctypes.CDLL(lib_path)




def native_stmvl(__py_matrix, __py_window, __py_gamma, __py_alpha):
    """
    Perform matrix imputation using the STMVL algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).
    __py_window : int
        The window size for the temporal component in the STMVL algorithm.
    __py_gamma : float
        The smoothing parameter for temporal weight (0 < gamma < 1).
    __py_alpha : float
        The power for the spatial weight.

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation.

    Notes
    -----
    The STMVL algorithm leverages temporal and spatial relationships to recover missing values in a matrix.
    The native C++ implementation is invoked for better performance.

    Example
    -------
    >>> imputed_data = stmvl(contamination=contamination_matrix, window_size=2, gamma=0.85, alpha=7)
    >>> print(imputed_data)

    """

    shared_lib = load_share_lib()

    __py_sizen = len(__py_matrix);
    __py_sizem = len(__py_matrix[0]);

    assert (__py_window >= 2);
    assert (__py_gamma > 0.0);
    assert (__py_gamma < 1.0);
    assert (__py_alpha > 0.0);

    __ctype_sizen = __native_c_types_import.c_ulonglong(__py_sizen);
    __ctype_sizem = __native_c_types_import.c_ulonglong(__py_sizem);

    __ctype_window = __native_c_types_import.c_ulonglong(__py_window);
    __ctype_gamma = __native_c_types_import.c_double(__py_gamma);
    __ctype_alpha = __native_c_types_import.c_double(__py_alpha);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_input_matrix = __marshal_as_native_column(__py_matrix);

    # extern "C" void
    # stmvl_imputation_parametrized(
    #         double *matrixNative, size_t dimN, size_t dimM,
    #         size_t window_size, double gamma, double alpha
    # )
    shared_lib.stmvl_imputation_parametrized(
        __ctype_input_matrix, __ctype_sizen, __ctype_sizem,
        __ctype_window, __ctype_gamma, __ctype_alpha
    );

    __py_recovered = __marshal_as_numpy_column(__ctype_input_matrix, __py_sizen, __py_sizem);

    return __py_recovered;


def stmvl(contamination, window_size, gamma, alpha, logs=True):
    """
    CDREC algorithm for imputation of missing data
    :author: Quentin Nater

    :param contamination: time series with contamination
    :param window_size: window size for temporal component
    :param gamma: smoothing parameter for temporal weight
    :param alpha: power for spatial weight

    :param logs: print logs of time execution

    :return: imputed_matrix, metrics : all time series with imputation data and their metrics

    """
    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    imputed_matrix = native_stmvl(contamination, window_size, gamma, alpha)

    end_time = time.time()
    if logs:
        print(f"\n\t\t> logs, imputation stvml - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return imputed_matrix


