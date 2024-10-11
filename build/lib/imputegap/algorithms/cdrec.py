import ctypes
import os
import platform
import time
import ctypes as __native_c_types_import;
import numpy as __numpy_import;


def __marshal_as_numpy_column(__ctype_container, __py_sizen, __py_sizem):
    __numpy_marshal = __numpy_import.array(__ctype_container).reshape(__py_sizem, __py_sizen).T;

    return __numpy_marshal;


def __marshal_as_native_column(__py_matrix):
    __py_input_flat = __numpy_import.ndarray.flatten(__py_matrix.T);
    __ctype_marshal = __numpy_import.ctypeslib.as_ctypes(__py_input_flat);

    return __ctype_marshal;


def load_share_lib(name="lib_cdrec"):
    """
    Determine the OS and load the correct shared library
    :param name: name of the library
    :return: the correct path to the library
    """

    local_path_win = './algorithms/lib/' + name + '.dll'
    local_path_lin = './algorithms/lib/' + name + '.so'

    if not os.path.exists(local_path_lin):
        local_path_win = './imputegap/algorithms/lib/' + name + '.dll'
        local_path_lin = './imputegap/algorithms/lib/' + name + '.so'

    if platform.system() == 'Windows':
        lib_path = os.path.join(local_path_win)
    else:
        lib_path = os.path.join(local_path_lin)

    return ctypes.CDLL(lib_path)


def native_cdrec(__py_matrix, __py_rank, __py_eps, __py_iters):
    """
    Recovers missing values (designated as NaN) in a matrix. Supports additional parameters
    :param __py_matrix: 2D array
    :param __py_rank: truncation rank to be used (0 = detect truncation automatically)
    :param __py_eps: threshold for difference during recovery
    :param __py_iters: maximum number of allowed iterations for the algorithms
    :return: 2D array recovered matrix
    """

    shared_lib = load_share_lib()

    __py_sizen = len(__py_matrix);
    __py_sizem = len(__py_matrix[0]);

    assert (__py_rank >= 0);
    assert (__py_rank < __py_sizem);
    assert (__py_eps > 0);
    assert (__py_iters > 0);

    __ctype_sizen = __native_c_types_import.c_ulonglong(__py_sizen);
    __ctype_sizem = __native_c_types_import.c_ulonglong(__py_sizem);

    __ctype_rank = __native_c_types_import.c_ulonglong(__py_rank);
    __ctype_eps = __native_c_types_import.c_double(__py_eps);
    __ctype_iters = __native_c_types_import.c_ulonglong(__py_iters);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_input_matrix = __marshal_as_native_column(__py_matrix);

    shared_lib.cdrec_imputation_parametrized(
        __ctype_input_matrix, __ctype_sizen, __ctype_sizem,
        __ctype_rank, __ctype_eps, __ctype_iters
    );

    __py_recovered = __marshal_as_numpy_column(__ctype_input_matrix, __py_sizen, __py_sizem);

    return __py_recovered;


def cdrec(contamination, truncation_rank, iterations, epsilon, logs=True):
    """
    CDREC algorithm for imputation of missing data
    @author : Quentin Nater

    :param contamination: time series with contamination
    :param truncation_rank: rank of reduction of the matrix (must be higher than 1 and smaller than the limit of series)
    :param epsilon : learning rate
    :param iterations : number of iterations

    :param logs: print logs of time execution

    :return: imputed_matrix, metrics : all time series with imputation data and their metrics

    """
    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    imputed_matrix = native_cdrec(contamination, truncation_rank, epsilon, iterations)

    end_time = time.time()

    if logs:
        print(f"\n\t\t> logs, imputation cdrec - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return imputed_matrix
