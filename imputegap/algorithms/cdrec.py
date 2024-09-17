import ctypes as __native_c_types_import;
import numpy as __numpy_import;


__NATIVE_CENTROID_LIBRARY_PATH_DEBUG = "./Wrapper/src/libAlgoCollection.so"; # same folder
__NATIVE_CENTROID_LIBRARY_PATH = "./Wrapper/libAlgoCollection.so"; # will pick up from anything in $PATH
__NATIVE_CENTROID_LIBRARY_PATH_ALT = "/Home/naterq/imputegap/imputegap/Wrapper/src/libAlgoCollection.so"; # manual
__NATIVE_CENTROID_LIBRARY_PATH_ALT_WSL = "/mnt/d/Git/imputegap/imputegap/Wrapper/src/libAlgoCollection.so"; # manual
__NATIVE_CENTROID_LIBRARY_PATH_ALT_WSL_LAPTOP = "/mnt/c/Git/imputegap/imputegap/Wrapper/src/libAlgoCollection.so"; # manual
__NATIVE_CENTROID_LIBRARY_PATH_Linux = "../Wrapper/src/libAlgoCollection.so"; # manual
__NATIVE_CENTROID_LIBRARY_PATH_Linux_2 = "../../Wrapper/src/libAlgoCollection.so"; # manual


paths = [
    __NATIVE_CENTROID_LIBRARY_PATH,
    __NATIVE_CENTROID_LIBRARY_PATH_ALT,
    __NATIVE_CENTROID_LIBRARY_PATH_ALT_WSL,
    __NATIVE_CENTROID_LIBRARY_PATH_ALT_WSL_LAPTOP,
    __NATIVE_CENTROID_LIBRARY_PATH_Linux,
    __NATIVE_CENTROID_LIBRARY_PATH_Linux_2,
    __NATIVE_CENTROID_LIBRARY_PATH_DEBUG
]


def __marshal_as_numpy_column(__ctype_container, __py_sizen, __py_sizem):
    __numpy_marshal = __numpy_import.array(__ctype_container).reshape(__py_sizem, __py_sizen).T;

    return __numpy_marshal;

def __marshal_as_native_column(__py_matrix):
    __py_input_flat = __numpy_import.ndarray.flatten(__py_matrix.T);
    __ctype_marshal = __numpy_import.ctypeslib.as_ctypes(__py_input_flat);

    return __ctype_marshal;


def native_cdrec_param(__py_matrix, __py_rank, __py_eps, __py_iters):
    """
    Recovers missing values (designated as NaN) in a matrix. Supports additional parameters
    :param __py_matrix: 2D array
    :param __py_rank: truncation rank to be used (0 = detect truncation automatically)
    :param __py_eps: threshold for difference during recovery
    :param __py_iters: maximum number of allowed iterations for the algorithms
    :return: 2D array recovered matrix
    """

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

    """
    for path in paths:
        if __os_path_import.isfile(path):
            __ctype_libcd_native = __native_c_types_import.cdll.LoadLibrary(path)
            break
    else:
        print("Cannot load the shared library - file not found")
        raise Exception('Failed to load the shared library.')


    __ctype_libcd_native.cdrec_imputation_simple(
        __ctype_input_matrix, __ctype_sizen, __ctype_sizem,
        __ctype_rank
    );
    """

    __py_recovered = __marshal_as_numpy_column(__ctype_input_matrix, __py_sizen, __py_sizem);

    return __py_recovered;

# end function