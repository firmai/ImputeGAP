import ctypes
import os
import toml
import importlib.resources


def display_title(title="Master Thesis", aut="Quentin Nater", lib="ImputeGAP", university="University Fribourg - exascale infolab"):
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
        The university or institution (default is "University Fribourg - exascale infolab").

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

    if set_name in ["bafu", "chlorine", "climate", "drift", "eeg", "eeg-test", "meteo", "test", "test-large"]:
        return set_name + ".txt"
    else:
        filepath = "../imputegap/dataset/" + set_name + ".txt"

        if not os.path.exists(filepath):
            filepath = filepath[1:]

        return filepath


def get_save_path_asset():
    """
    Find the accurate path for saving test files.

    Returns
    -------
    str
        The correct file path for saving test files.
    """
    filepath = "../tests/assets"

    if not os.path.exists(filepath):
        filepath = filepath[1:]

    return filepath


def load_parameters(query: str = "default", algorithm: str = "cdrec", dataset: str = "chlorine", optimizer: str="b", path=None):
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
    filepath = ""
    if query == "default":

        if path is None:
            filepath = importlib.resources.files('imputegap.env').joinpath("./default_values.toml")
        else:
            filepath = path

    elif query == "optimal":

        if path is None:
            filename = "./optimal_parameters_"+str(optimizer)+"_"+str(dataset)+"_"+str(algorithm)+".toml"
            filepath = importlib.resources.files('imputegap.params').joinpath(filename)
        else:
            filepath = path

    else:
        print("Query not found for this function ('optimal' or 'default')")

    if not os.path.exists(filepath):
        filepath = filepath[1:]

    with open(filepath, "r") as _:
        config = toml.load(filepath)

    if algorithm == "cdrec":
        truncation_rank = int(config['cdrec']['rank'])
        epsilon = config['cdrec']['epsilon']
        iterations = int(config['cdrec']['iteration'])
        return (truncation_rank, float(epsilon), iterations)
    elif algorithm == "stmvl":
        window_size = int(config['stmvl']['window_size'])
        gamma = float(config['stmvl']['gamma'])
        alpha = int(config['stmvl']['alpha'])
        return (window_size, gamma, alpha)
    elif algorithm == "iim":
        learning_neighbors = int(config['iim']['learning_neighbors'])
        if query == "default":
            algo_code = config['iim']['algorithm_code']
            return (learning_neighbors, algo_code)
        else:
            return (learning_neighbors,)
    elif algorithm == "mrnn":
        hidden_dim = int(config['mrnn']['hidden_dim'])
        learning_rate = float(config['mrnn']['learning_rate'])
        iterations = int(config['mrnn']['iterations'])
        if query == "default":
            sequence_length = int(config['mrnn']['sequence_length'])
            return (hidden_dim, learning_rate, iterations, sequence_length)
        else:
            return (hidden_dim, learning_rate, iterations)
    elif algorithm == "greedy":
        n_calls = int(config['greedy']['n_calls'])
        selected_metrics = config['greedy']['selected_metrics']
        return (n_calls, [selected_metrics])
    elif algorithm == "bayesian":
        n_calls = int(config['bayesian']['n_calls'])
        n_random_starts = int(config['bayesian']['n_random_starts'])
        acq_func = str(config['bayesian']['acq_func'])
        selected_metrics = config['bayesian']['selected_metrics']
        return (n_calls, n_random_starts, acq_func, [selected_metrics])
    elif algorithm == "pso":
        n_particles = int(config['pso']['n_particles'])
        c1 = float(config['pso']['c1'])
        c2 = float(config['pso']['c2'])
        w = float(config['pso']['w'])
        iterations = int(config['pso']['iterations'])
        n_processes = int(config['pso']['n_processes'])
        selected_metrics = config['pso']['selected_metrics']
        return (n_particles, c1, c2, w, iterations, n_processes, [selected_metrics])
    elif algorithm == "sh":
        num_configs = int(config['sh']['num_configs'])
        num_iterations = int(config['sh']['num_iterations'])
        reduction_factor = int(config['sh']['reduction_factor'])
        selected_metrics = config['sh']['selected_metrics']
        return (num_configs, num_iterations, reduction_factor, [selected_metrics])
    elif algorithm == "colors":
        colors = config['colors']['plot']
        return colors
    else :
        print("Default/Optimal config not found for this algorithm")
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
        print("The percentage", percentage, "is out of the acceptable range", low_limit, "-", high_limit, ".")
        return percentage

def format_selection(ts, selection):
    """
    Format the selection of time series based on keywords provided by the user.

    Parameters
    ----------
    ts : numpy.ndarray
        The dataset of time series to be contaminated.
    selection : list
        The list of selected series to be formatted. It can include:
        - "*" to select all series.
        - "-" followed by an integer to exclude that many series from the end.
        - "+" followed by an integer to include series from that starting point onward.

    Returns
    -------
    list
        A list of formatted series indices.

    Raises
    ------
    ValueError
        If the selection format is invalid.

    Notes
    -----
    - If selection is ["*"], all series in `ts` are selected.
    - If selection is ["-X"], all but the last X series are selected.
    - If selection is ["+X"], all series starting from index X are selected.
    """
    if not selection:
        selection = ["*"]

    if selection == ["*"]:
        series_selected = []
        for i in range(0, ts.shape[0]):
            series_selected.append(str(i))
        return series_selected

    elif "-" in selection[0]:
        series_selected = []
        value = selection[0]
        ending = int(value[1:])
        for i in range(0, ts.shape[0] - ending):
            series_selected.append(str(i))
        return series_selected

    elif "+" in selection[0]:
        series_selected = []
        value = selection[0]
        starting = int(value[1:])
        for i in range(starting, ts.shape[0]):
            series_selected.append(str(i))
        return series_selected

    else:
        return selection

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

    return ctypes.CDLL(lib_path)