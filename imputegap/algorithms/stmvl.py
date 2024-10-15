import ctypes
import time
import os
import importlib.resources
import ctypes as __native_c_types_import;
import numpy as __numpy_import;

def __marshal_as_numpy_column(__ctype_container, __py_sizen, __py_sizem):
    __numpy_marshal = __numpy_import.array(__ctype_container).reshape(__py_sizem, __py_sizen).T;

    return __numpy_marshal;

def __marshal_as_native_column(__py_matrix):
    __py_input_flat = __numpy_import.ndarray.flatten(__py_matrix.T);
    __ctype_marshal = __numpy_import.ctypeslib.as_ctypes(__py_input_flat);

    return __ctype_marshal;


def load_share_lib(name = "lib_stmvl", lib=True):
    """
    Determine the OS and load the correct shared library
    :param name: name of the library
    :return: the correct path to the library
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
    Recovers missing values (designated as NaN) in a matrix. Supports additional parameters
    :param __py_matrix: 2D array
    :param __py_window: window size for temporal component
    :param __py_gamma: smoothing parameter for temporal weight
    :param __py_alpha: power for spatial weight
    :return: 2D array recovered matrix
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


