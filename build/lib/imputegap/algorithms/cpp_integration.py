import ctypes
import time
import ctypes as __native_c_types_import;
import numpy as __numpy_import;

from imputegap.tools import utils

# =========================================================== #
# IN CASE OF NEED, YOU CAN ADAPT AND TAKE CDREC.PY AS A MODEL #
# =========================================================== #

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


def native_algo(__py_matrix, __py_param):
    """
    Perform matrix imputation using the CDRec algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).
    __py_param : int
        parameters to adapt

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation.

    """

    shared_lib = utils.load_share_lib("to_adapt.so")

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);


    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    # depends on your needs
    __py_param = __native_c_types_import.c_ulonglong(__py_param);
    __py_param = __native_c_types_import.c_double(__py_param);
    __py_param = __native_c_types_import.c_ulonglong(__py_param);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = __marshal_as_native_column(__py_matrix);

    # call your algorithm
    shared_lib.your_algo_name(__ctype_matrix, __ctype_size_n, __ctype_size_m, __py_param);

    # convert back to numpy
    __py_imputed_matrix = __marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def your_algo(contamination, param, logs=True):
    """
    CDRec algorithm for matrix imputation of missing values using Centroid Decomposition.

    Parameters
    ----------
    contamination : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    param : to adapt
        to adapt
    logs : bool, optional
        Whether to log the execution time (default is True).
    lib_path : str, optional
        Custom path to the shared library file (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.
    """
    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    imputed_matrix = native_algo(contamination, param)

    end_time = time.time()

    if logs:
        print(f"\n\t\t> logs, imputation algo - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return imputed_matrix
