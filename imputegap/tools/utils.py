import ctypes
import os
import toml
import importlib.resources
import numpy as __numpy_import;


def config_impute_algorithm(incomp_data, algorithm):
    """
    Configure and execute algorithm for selected imputation imputer and pattern.

    Parameters
    ----------
    incomp_data : TimeSeries
        TimeSeries object containing dataset.
    algorithm : str
        Name of algorithm

    Returns
    -------
    BaseImputer
        Configured imputer instance with optimal parameters.
    """

    from imputegap.recovery.imputation import Imputation

    # 1st generation
    if algorithm == "cdrec":
        imputer = Imputation.MatrixCompletion.CDRec(incomp_data)
    elif algorithm == "stmvl":
        imputer = Imputation.PatternSearch.STMVL(incomp_data)
    elif algorithm == "iim":
        imputer = Imputation.Statistics.IIM(incomp_data)
    elif algorithm == "mrnn":
        imputer = Imputation.DeepLearning.MRNN(incomp_data)

    # 2nd generation
    elif algorithm == "iterative_svd":
        imputer = Imputation.MatrixCompletion.IterativeSVD(incomp_data)
    elif algorithm == "grouse":
        imputer = Imputation.MatrixCompletion.GROUSE(incomp_data)
    elif algorithm == "dynammo":
        imputer = Imputation.PatternSearch.DynaMMo(incomp_data)
    elif algorithm == "rosl":
        imputer = Imputation.MatrixCompletion.ROSL(incomp_data)
    elif algorithm == "soft_impute":
        imputer = Imputation.MatrixCompletion.SoftImpute(incomp_data)
    elif algorithm == "spirit":
        imputer = Imputation.MatrixCompletion.SPIRIT(incomp_data)
    elif algorithm == "svt":
        imputer = Imputation.MatrixCompletion.SVT(incomp_data)
    elif algorithm == "tkcm":
        imputer = Imputation.PatternSearch.TKCM(incomp_data)
    elif algorithm == "deep_mvi":
        imputer = Imputation.DeepLearning.DeepMVI(incomp_data)
    elif algorithm == "brits":
        imputer = Imputation.DeepLearning.BRITS(incomp_data)
    elif algorithm == "mpin":
        imputer = Imputation.DeepLearning.MPIN(incomp_data)
    elif algorithm == "pristi":
        imputer = Imputation.DeepLearning.PRISTI(incomp_data)
    else:
        imputer = Imputation.Statistics.MeanImpute(incomp_data)

    return imputer


def config_contamination(ts, pattern, dataset_rate=0.4, series_rate=0.4, block_size=10, offset=0.1, seed=True, limit=1, shift=0.05, std_dev=0, explainer=False):
    """
    Configure and execute contamination for selected imputation algorithm and pattern.

    Parameters
    ----------
    rate : float
        Mean parameter for contamination missing percentage rate.
    ts_test : TimeSeries
        A TimeSeries object containing dataset.
    pattern : str
        Type of contamination pattern (e.g., "mcar", "mp", "blackout", "disjoint", "overlap", "gaussian").
    block_size_mcar : int
        Size of blocks removed in MCAR

    Returns
    -------
    TimeSeries
        TimeSeries object containing contaminated data.
    """
    if pattern == "mcar":
        incomp_data = ts.Contamination.mcar(input_data=ts.data, dataset_rate=dataset_rate, series_rate=series_rate, block_size=block_size, offset=offset, seed=seed, explainer=explainer)
    elif pattern == "mp":
        incomp_data = ts.Contamination.missing_percentage(input_data=ts.data, dataset_rate=dataset_rate, series_rate=series_rate, offset=offset)
    elif pattern == "disjoint":
        incomp_data = ts.Contamination.disjoint(input_data=ts.data, series_rate=dataset_rate, limit=1, offset=offset)
    elif pattern == "overlap":
        incomp_data = ts.Contamination.overlap(input_data=ts.data, series_rate=dataset_rate, limit=limit, shift=shift, offset=offset)
    elif pattern == "gaussian":
        incomp_data = ts.Contamination.gaussian(input_data=ts.data, dataset_rate=dataset_rate, series_rate=series_rate, std_dev=std_dev, offset=offset, seed=True)
    else:
        incomp_data = ts.Contamination.blackout(input_data=ts.data, series_rate=dataset_rate, offset=offset)

    return incomp_data

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


def display_title(title="Master Thesis", aut="Quentin Nater", lib="ImputeGAP", university="University Fribourg"):
    """
    Display the title and author information.

    Parameters
    ----------
    title : str, optional
        The title of the thesis (default is "Master Thesis").
    aut : str, optional
        The author's name (default is "Quentin Nater").
    lib : str, optional
        The library or project name (default is "ImputeGAP").
    university : str, optional
        The university or institution (default is "University Fribourg").

    Returns
    -------
    None
    """

    print("=" * 100)
    print(f"{title} : {aut}")
    print("=" * 100)
    print(f"    {lib} - {university}")
    print("=" * 100)


def search_path(set_name="test"):
    """
    Find the accurate path for loading test files.

    Parameters
    ----------
    set_name : str, optional
        Name of the dataset (default is "test").

    Returns
    -------
    str
        The correct file path for the dataset.
    """

    if set_name in ["bafu", "chlorine", "climate", "drift", "eeg-reading", "eeg-alcohol", "fmri-objectviewing", "fmri-stoptask", "meteo", "test", "test-large"]:
        return set_name + ".txt"
    else:
        filepath = "../imputegap/dataset/" + set_name + ".txt"

        if not os.path.exists(filepath):
            filepath = filepath[1:]
        return filepath


def load_parameters(query: str = "default", algorithm: str = "cdrec", dataset: str = "chlorine", optimizer: str = "b", path=None):
    """
    Load default or optimal parameters for algorithms from a TOML file.

    Parameters
    ----------
    query : str, optional
        'default' or 'optimal' to load default or optimal parameters (default is "default").
    algorithm : str, optional
        Algorithm to load parameters for (default is "cdrec").
    dataset : str, optional
        Name of the dataset (default is "chlorine").
    optimizer : str, optional
        Optimizer type for optimal parameters (default is "b").
    path : str, optional
        Custom file path for the TOML file (default is None).

    Returns
    -------
    tuple
        A tuple containing the loaded parameters for the given algorithm.
    """
    if query == "default":
        if path is None:
            filepath = importlib.resources.files('imputegap.env').joinpath("./default_values.toml")
            if not filepath.is_file():
                filepath = "./env/default_values.toml"
        else:
            filepath = path
            if not os.path.exists(filepath):
                filepath = "./env/default_values.toml"

    elif query == "optimal":
        if path is None:
            filename = "./optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"
            filepath = importlib.resources.files('imputegap.params').joinpath(filename)
            if not filepath.is_file():
                filepath = "./params/optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"
        else:
            filepath = path
            if not os.path.exists(filepath):
                filepath = "./params/optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"

    else:
        filepath = None
        print("Query not found for this function ('optimal' or 'default')")

    if not os.path.exists(filepath):
        filepath = "./params/optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"
        if not os.path.exists(filepath):
            filepath = filepath[1:]

    with open(filepath, "r") as _:
        config = toml.load(filepath)

    print("\n\t\t\t\t(SYS) Inner files loaded : ", filepath, "\n")

    if algorithm == "cdrec":
        truncation_rank = int(config[algorithm]['rank'])
        epsilon = float(config[algorithm]['epsilon'])
        iterations = int(config[algorithm]['iteration'])
        return (truncation_rank, epsilon, iterations)
    elif algorithm == "stmvl":
        window_size = int(config[algorithm]['window_size'])
        gamma = float(config[algorithm]['gamma'])
        alpha = int(config[algorithm]['alpha'])
        return (window_size, gamma, alpha)
    elif algorithm == "iim":
        learning_neighbors = int(config[algorithm]['learning_neighbors'])
        if query == "default":
            algo_code = config[algorithm]['algorithm_code']
            return (learning_neighbors, algo_code)
        else:
            return (learning_neighbors,)
    elif algorithm == "mrnn":
        hidden_dim = int(config[algorithm]['hidden_dim'])
        learning_rate = float(config[algorithm]['learning_rate'])
        iterations = int(config[algorithm]['iterations'])
        if query == "default":
            sequence_length = int(config[algorithm]['sequence_length'])
            return (hidden_dim, learning_rate, iterations, sequence_length)
        else:
            return (hidden_dim, learning_rate, iterations)
    elif algorithm == "iterative_svd":
        truncation_rank = int(config[algorithm]['rank'])
        return (truncation_rank)
    elif algorithm == "grouse":
        max_rank = int(config[algorithm]['max_rank'])
        return (max_rank)
    elif algorithm == "dynammo":
        h = int(config[algorithm]['h'])
        max_iteration = int(config[algorithm]['max_iteration'])
        approximation = bool(config[algorithm]['approximation'])
        return (h, max_iteration, approximation)
    elif algorithm == "rosl":
        rank = int(config[algorithm]['rank'])
        regularization = int(config[algorithm]['regularization'])
        return (rank, regularization)
    elif algorithm == "soft_impute":
        max_rank = int(config[algorithm]['max_rank'])
        return (max_rank)
    elif algorithm == "spirit":
        k = int(config[algorithm]['k'])
        w = int(config[algorithm]['w'])
        lvalue = float(config[algorithm]['lvalue'])
        return (k, w, lvalue)
    elif algorithm == "svt":
        tau = float(config[algorithm]['tau'])
        return (tau)
    elif algorithm == "tkcm":
        rank = int(config[algorithm]['rank'])
        return (rank)
    elif algorithm == "deep_mvi":
        max_epoch = int(config[algorithm]['max_epoch'])
        patience = int(config[algorithm]['patience'])
        return (max_epoch, patience)
    elif algorithm == "brits":
        model = str(config[algorithm]['model'])
        epoch = int(config[algorithm]['epoch'])
        batch_size = int(config[algorithm]['batch_size'])
        nbr_features = int(config[algorithm]['nbr_features'])
        hidden_layers = int(config[algorithm]['hidden_layers'])
        return (model, epoch, batch_size, nbr_features, hidden_layers)
    elif algorithm == "mpin":
        incre_mode = str(config[algorithm]['incre_mode'])
        window = int(config[algorithm]['window'])
        k = int(config[algorithm]['k'])
        learning_rate = float(config[algorithm]['learning_rate'])
        weight_decay = float(config[algorithm]['weight_decay'])
        epochs = int(config[algorithm]['epochs'])
        threshold = float(config[algorithm]['threshold'])
        base = str(config[algorithm]['base'])
        return (incre_mode, window, k, learning_rate, weight_decay, epochs, threshold, base)
    elif algorithm == "pristi":
        target_strategy = str(config[algorithm]['target_strategy'])
        unconditional = bool(config[algorithm]['unconditional'])
        seed = int(config[algorithm]['seed'])
        device = str(config[algorithm]['device'])
        return (target_strategy, unconditional, seed, device)
    elif algorithm == "greedy":
        n_calls = int(config[algorithm]['n_calls'])
        metrics = config[algorithm]['metrics']
        return (n_calls, [metrics])
    elif algorithm == "bayesian":
        n_calls = int(config[algorithm]['n_calls'])
        n_random_starts = int(config[algorithm]['n_random_starts'])
        acq_func = str(config[algorithm]['acq_func'])
        metrics = config['bayesian']['metrics']
        return (n_calls, n_random_starts, acq_func, [metrics])
    elif algorithm == "pso":
        n_particles = int(config[algorithm]['n_particles'])
        c1 = float(config[algorithm]['c1'])
        c2 = float(config[algorithm]['c2'])
        w = float(config[algorithm]['w'])
        iterations = int(config[algorithm]['iterations'])
        n_processes = int(config[algorithm]['n_processes'])
        metrics = config[algorithm]['metrics']
        return (n_particles, c1, c2, w, iterations, n_processes, [metrics])
    elif algorithm == "sh":
        num_configs = int(config[algorithm]['num_configs'])
        num_iterations = int(config[algorithm]['num_iterations'])
        reduction_factor = int(config[algorithm]['reduction_factor'])
        metrics = config[algorithm]['metrics']
        return (num_configs, num_iterations, reduction_factor, [metrics])
    elif algorithm == "forecaster-naive":
        strategy = str(config[algorithm]['strategy'])
        window_length = int(config[algorithm]['window_length'])
        sp = int(config[algorithm]['sp'])
        return {"strategy": strategy, "window_length": window_length, "sp": sp}
    elif algorithm == "forecaster-exp-smoothing":
        trend = str(config[algorithm]['trend'])
        seasonal = str(config[algorithm]['seasonal'])
        sp = int(config[algorithm]['sp'])
        return {"trend": trend, "seasonal": seasonal, "sp": sp}
    elif algorithm == "forecaster-prophet":
        seasonality_mode = str(config[algorithm]['seasonality_mode'])
        n_changepoints = int(config[algorithm]['n_changepoints'])
        return {"seasonality_mode": seasonality_mode, "n_changepoints": n_changepoints}
    elif algorithm == "colors":
        colors = config[algorithm]['plot']
        return colors
    elif algorithm == "other":
        return config
    else:
        print("\t\t(SYS) Default/Optimal config not found for this algorithm")
        return None


def verification_limitation(percentage, low_limit=0.01, high_limit=1.0):
    """
    Format and verify that the percentage given by the user is within acceptable bounds.

    Parameters
    ----------
    percentage : float
        The percentage value to be checked and potentially adjusted.
    low_limit : float, optional
        The lower limit of the acceptable percentage range (default is 0.01).
    high_limit : float, optional
        The upper limit of the acceptable percentage range (default is 1.0).

    Returns
    -------
    float
        Adjusted percentage based on the limits.

    Raises
    ------
    ValueError
        If the percentage is outside the accepted limits.

    Notes
    -----
    - If the percentage is between 1 and 100, it will be divided by 100 to convert it to a decimal format.
    - If the percentage is outside the low and high limits, the function will print a warning and return the original value.
    """
    if low_limit <= percentage <= high_limit:
        return percentage  # No modification needed

    elif 1 <= percentage <= 100:
        print(f"The percentage {percentage} is between 1 and 100. Dividing by 100 to convert to a decimal.")
        return percentage / 100

    else:
        raise ValueError("The percentage is out of the acceptable range.")


def load_share_lib(name="lib_cdrec", lib=True):
    """
    Load the shared library based on the operating system.

    Parameters
    ----------
    name : str, optional
        The name of the shared library (default is "lib_cdrec").
    lib : bool, optional
        If True, the function loads the library from the default 'imputegap' path; if False, it loads from a local path (default is True).

    Returns
    -------
    ctypes.CDLL
        The loaded shared library object.
    """

    if lib:
        lib_path = importlib.resources.files('imputegap.algorithms.lib').joinpath("./" + str(name))
    else:
        local_path_lin = './algorithms/lib/' + name + '.so'

        if not os.path.exists(local_path_lin):
            local_path_lin = './imputegap/algorithms/lib/' + name + '.so'

        lib_path = os.path.join(local_path_lin)
        print("\t\t(SYS) lib loaded from:", lib_path)


    return ctypes.CDLL(lib_path)


def save_optimization(optimal_params, algorithm="cdrec", dataset="", optimizer="b", file_name=None):
    """
    Save the optimization parameters to a TOML file for later use without recomputing.

    Parameters
    ----------
    optimal_params : dict
        Dictionary of the optimal parameters.
    algorithm : str, optional
        The name of the imputation algorithm (default is 'cdrec').
    dataset : str, optional
        The name of the dataset (default is an empty string).
    optimizer : str, optional
        The name of the optimizer used (default is 'b').
    file_name : str, optional
        The name of the TOML file to save the results (default is None).

    Returns
    -------
    None
    """
    if file_name is None:
        file_name = "../params/optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"

    if not os.path.exists(file_name):
        file_name = file_name[1:]

    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if algorithm == "mrnn":
        params_to_save = {
            algorithm: {
                "hidden_dim": int(optimal_params[0]),
                "learning_rate": optimal_params[1],
                "iterations": int(optimal_params[2])
            }
        }
    elif algorithm == "stmvl":
        params_to_save = {
            algorithm: {
                "window_size": int(optimal_params[0]),
                "gamma": optimal_params[1],
                "alpha": int(optimal_params[2])
            }
        }
    elif algorithm == "iim":
        params_to_save = {
            algorithm: {
                "learning_neighbors": int(optimal_params[0])
            }
        }
    else:
        params_to_save = {
            algorithm: {
                "rank": int(optimal_params[0]),
                "epsilon": optimal_params[1],
                "iteration": int(optimal_params[2])
            }
        }

    try:
        with open(file_name, 'w') as file:
            toml.dump(params_to_save, file)
        print(f"\n\t\t(SYS) Optimization parameters successfully saved to {file_name}")
    except Exception as e:
        print(f"\n\t\t(SYS) An error occurred while saving the file: {e}")
