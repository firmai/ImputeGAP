import os
import time
import numpy as np
import matplotlib
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import importlib.resources

from imputegap.tools import utils

# Use Agg backend if in a headless or CI environment
if os.getenv('DISPLAY') is None or os.getenv('CI') is not None:
    matplotlib.use("Agg")
    print("Running in a headless environment or CI. Using Agg backend.")
else:
    try:
        # Attempt to use TkAgg if a display is available and we're not in CI
        matplotlib.use("TkAgg")
        if importlib.util.find_spec("tkinter") is not None:
            print("tkinter is available.")
        else:
            print("tkinter is not available.")
        print("Using TkAgg backend with tkinter support.")
    except (ImportError, RuntimeError):
        # Fallback to Agg if TkAgg is unavailable
        matplotlib.use("Agg")
        print("TkAgg is unavailable. Using Agg backend.")

from matplotlib import pyplot as plt  # type: ignore


class TimeSeries:
    """
    Class for managing and manipulating time series data.

    This class allows importing, normalizing, and visualizing time series datasets. It also provides methods
    to contaminate the datasets with missing values and plot results.

    Methods
    -------
    __init__() :
        Initializes the TimeSeries object.

    import_matrix(data=None) :
        Imports a matrix of time series data.

    load_timeseries(data=None, max_series=None, max_values=None, header=False) :
        Loads time series data from a file or predefined dataset.

    print(limit=10, view_by_series=False) :
        Prints a limited number of time series from the dataset.

    print_results(metrics, algorithm="") :
        Prints the results of the imputation process.

    normalize(normalizer="z_score") :
        Normalizes the time series dataset.

    plot(raw_data, infected_data=None, imputed_data=None, title="Time Series Data",
         max_series=None, max_values=None, size=(16, 8), save_path="", display=True) :
        Plots the time series data, including raw, contaminated, or imputed data.

    Contaminate :
        Class containing methods to contaminate time series data with missing values based on different scenarios.

    """

    def __init__(self):
        """
        Initialize the TimeSeries object.

        The class works with time series datasets, where each series is separated by space, and values
        are separated by newline characters.

        IMPORT FORMAT : (Values,Series) : series are seperated by "SPACE" et values by "\\n"
        """
        self.data = None

    def import_matrix(self, data=None):
        """"
        Imports a matrix of time series data.

        The data can be provided as a list or a NumPy array. The format is (Series, Values),
        where series are separated by space, and values are separated by newline characters.

        Parameters
        ----------
        data : list or numpy.ndarray, optional
            The matrix of time series data to import.

        Returns
        -------
        TimeSeries
            The TimeSeries object with the imported data.
        """
        if data is not None:
            if isinstance(data, list):
                self.data = np.array(data)

            elif isinstance(data, np.ndarray):
                self.data = data
            else:
                print("\nThe time series has not been loaded, format unknown\n")
                self.data = None
                raise ValueError("Invalid input for import_matrix")

            return self

    def load_timeseries(self, data, max_series=None, max_values=None, header=False):
        """
        Loads time series data from a file or predefined dataset.

        The data is loaded as a matrix of shape (Values, Series). You can limit the number of series
        or values per series for computational efficiency.

        Parameters
        ----------
        data : str
            The file path or name of a predefined dataset (e.g., 'bafu.txt').
        max_series : int, optional
            The maximum number of series to load.
        max_values : int, optional
            The maximum number of values per series.
        header : bool, optional
            Whether the dataset has a header. Default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries object with the loaded data.
        """

        if data is not None:
            if isinstance(data, str):
                print("\nThe time series has been loaded from " + str(data) + "\n")

                #  update path form inner library datasets
                if data in ["bafu.txt", "chlorine.txt", "climate.txt", "drift.txt", "eeg-alcohol.txt", "eeg-reading.txt",
                            "meteo.txt", "test.txt", "test-large.txt"]:
                    data = importlib.resources.files('imputegap.dataset').joinpath(data)

                self.data = np.genfromtxt(data, delimiter=' ', max_rows=max_values, skip_header=int(header))

                if max_series is not None:
                    self.data = self.data[:, :max_series]
            else:
                print("\nThe time series has not been loaded, format unknown\n")
                self.data = None
                raise ValueError("Invalid input for load_timeseries")

            self.data = self.data.T

            return self

    def print(self, limit=10, view_by_series=False):
        """
        Prints a limited number of time series from the dataset.

        Parameters
        ----------
        limit : int, optional
            The number of series to print. Default is 10.
        view_by_series : bool, optional
            Whether to view by series (True) or by values (False).

        Returns
        -------
        None
        """

        print("\nTime Series set :")

        to_print = self.data
        nbr_series, nbr_values = to_print.shape
        print_col, print_row = "Values", "Series"

        if not view_by_series:
            to_print = to_print.T
            print_col, print_row = "Series", "Values"

        print(f"{' ':19}", end="")
        for i, _ in enumerate(to_print[1]):
            if i < 10:
                print(f"{print_col} {i}", end=" " * 8)
            elif i < 100:
                print(f"{print_col} {i}", end=" " * 7)
            else:
                print(f"{print_col} {i}", end=" " * 6)
        print()

        for i, series in enumerate(to_print[:limit]):
            print(f"{print_row} {i} \t\t" + " ".join([f"{elem:15.10f}" for elem in series]))

        if limit < to_print.shape[0]:
            print("...")

        print("\nshape of the time series :", to_print.shape, "\n\tnumber of series =", nbr_series,
              "\n\tnumber of values =", nbr_values, "\n\n")

    def print_results(self, metrics, algorithm=""):
        """
        Prints the results of the imputation process.

        Parameters
        ----------
        metrics : dict
           A dictionary containing the imputation metrics to display.
        algorithm : str, optional
           The name of the algorithm used for imputation.

        Returns
        -------
        None
        """

        print("\n\nResults of the imputation ", algorithm, " :")
        for key, value in metrics.items():
            print(f"{key:<20} = {value}")
        print("\n")

    def normalize(self, normalizer="z_score"):
        """
        Normalize the time series dataset.

        Supported normalization techniques are "z_score" and "min_max". The method also logs
        the execution time for the normalization process.

        Parameters
        ----------
        normalizer : str, optional
            The normalization technique to use. Options are "z_score" or "min_max". Default is "z_score".

        Returns
        -------
        numpy.ndarray
            The normalized time series data.
        """
        print("Normalization of the original time series dataset with ", normalizer)
        self.data = self.data.T

        if normalizer == "min_max":
            start_time = time.time()  # Record start time

            # Compute the min and max for each series (column-wise), ignoring NaN
            ts_min = np.nanmin(self.data, axis=0)
            ts_max = np.nanmax(self.data, axis=0)

            # Compute the range for each series, and handle cases where the range is 0
            range_ts = ts_max - ts_min
            range_ts[range_ts == 0] = 1  # Prevent division by zero for constant series

            # Apply min-max normalization
            self.data = (self.data - ts_min) / range_ts

            end_time = time.time()
        elif normalizer == "z_lib":
            start_time = time.time()  # Record start time

            self.data = zscore(self.data, axis=0)

            end_time = time.time()

        elif normalizer == "m_lib":
            start_time = time.time()  # Record start time

            scaler = MinMaxScaler()
            self.data = scaler.fit_transform(self.data)

            end_time = time.time()
        else:
            start_time = time.time()  # Record start time

            mean = np.mean(self.data, axis=0)
            std_dev = np.std(self.data, axis=0)

            # Avoid division by zero: set std_dev to 1 where it is zero
            std_dev[std_dev == 0] = 1

            # Apply z-score normalization
            self.data = (self.data - mean) / std_dev

            end_time = time.time()

        self.data = self.data.T

        print(f"\n\t\t> logs, normalization {normalizer} - Execution Time: {(end_time - start_time):.4f} seconds\n")

    def plot(self, raw_data, infected_data=None, imputed_data=None, title="Time Series Data", max_series=None,
             max_values=None, series_x=None, size=(16, 8), save_path="", display=True):
        """
        Plot the time series data, including raw, contaminated, or imputed data.

        Parameters
        ----------
        raw_data : numpy.ndarray
            The original time series data without contamination.
        infected_data : numpy.ndarray, optional
            The contaminated time series data.
        imputed_data : numpy.ndarray, optional
            The imputed time series data.
        title : str, optional
            Title of the plot. Default is "Time Series Data".
        max_series : int, optional
            The maximum number of series to plot.
        max_values : int, optional
            The maximum number of values per series to plot.
        series_x : int, optional
            The index of a specific series to plot. If set, only this series will be plotted.
        size : tuple, optional
            Size of the plot in inches. Default is (16, 8).
        save_path : str, optional
            Path to save the plot locally.
        display : bool, optional
            Whether to display the plot. Default is True.

        Returns
        -------
        str or None
            The file path of the saved plot, if applicable.
        """
        number_of_series = 0
        plt.figure(figsize=size)
        plt.grid(True, linestyle='--', color='#d3d3d3', linewidth=0.6)

        if max_series is None:
            max_series, _ = raw_data.shape
        if max_values is None:
            _, max_values = raw_data.shape

        if raw_data is not None:

            colors = utils.load_parameters("default", algorithm="colors")

            # Determine range of series to plot
            series_indices = [series_x] if series_x is not None else range(raw_data.shape[0])

            for i in series_indices:
                color = colors[i % len(colors)]

                if infected_data is None and imputed_data is None:  # plot only raw matrix
                    plt.plot(np.arange(min(raw_data.shape[1], max_values)), raw_data[i, :max_values], linewidth=2.5,
                             color=color, linestyle='-', label=f'TS {i + 1}')

                if infected_data is not None and imputed_data is None:  # plot infected matrix
                    if np.isnan(infected_data[i, :]).any():
                        plt.plot(np.arange(min(raw_data.shape[1], max_values)), raw_data[i, :max_values], linewidth=1.5,
                                 color='r', linestyle='--', label=f'TS-MB {i + 1}')

                    plt.plot(np.arange(min(infected_data.shape[1], max_values)), infected_data[i, :max_values],
                             color=color, linewidth=2.5, linestyle='-', label=f'TS-RAW {i + 1}')

                if imputed_data is not None:  # plot imputed matrix
                    if np.isnan(infected_data[i, :]).any():
                        plt.plot(np.arange(min(imputed_data.shape[1], max_values)), imputed_data[i, :max_values],
                                 linestyle='-', color="r", label=f'TS-IMP {i + 1}')

                    if np.isnan(infected_data[i, :]).any():
                        plt.plot(np.arange(min(raw_data.shape[1], max_values)), raw_data[i, :max_values], linewidth=1.5,
                                 linestyle='--', color=color, label=f'TS-MB {i + 1}')

                    plt.plot(np.arange(min(infected_data.shape[1], max_values)), infected_data[i, :max_values],
                             color=color, linewidth=2.5, linestyle='-', label=f'TS-RAW {i + 1}')

                number_of_series += 1
                if number_of_series == max_series:
                    break

        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.title(title)
        plt.legend(
            loc='upper left',
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True,
            borderpad=1.5,
            bbox_to_anchor=(1.02, 1),  # Adjusted to keep the legend inside the window
        )

        file_path = None
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path + "/" + title.replace(" ", "") + "_graph.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            print("plots saved in ", file_path)

        if display:
            plt.show()

        # plt.close()

        return file_path

    class Contaminate:
        """
        Inner class to apply contamination scenarios to the time series data.

        Methods
        -------
        mcar(ts, series_impacted=0.2, missing_rate=0.2, block_size=10, protection=0.1, use_seed=True, seed=42, explainer=False) :
            Apply Missing Completely at Random (MCAR) contamination to the time series data.

        missing_percentage(ts, series_impacted=0.2, missing_rate=0.2, protection=0.1) :
            Apply missing percentage contamination to the time series data.

        blackout(ts, missing_rate=0.2, protection=0.1) :
            Apply blackout contamination to the time series data.
        """

        def mcar(ts, series_impacted=0.2, missing_rate=0.2, block_size=10, protection=0.1, use_seed=True, seed=42,
                 explainer=False):
            """
            Apply Missing Completely at Random (MCAR) contamination to the time series data.

            Parameters
            ----------
            ts : numpy.ndarray
                The time series dataset to contaminate.
            series_impacted : float, optional
                Percentage of series to contaminate (default is 0.2).
            missing_rate : float, optional
                Percentage of missing values per series (default is 0.2).
            block_size : int, optional
                Size of the block of missing data (default is 10).
            protection : float, optional
                Size of the uncontaminated section at the beginning of the series (default is 0.1).
            use_seed : bool, optional
                Whether to use a seed for reproducibility (default is True).
            seed : int, optional
                The value of the seed (default is 42).
            explainer : bool, optional
                Whether to apply MCAR to specific series for explanation purposes (default is False).

            Returns
            -------
            numpy.ndarray
                The contaminated time series data.
            """

            if use_seed:
                np.random.seed(seed)

            ts_contaminated = ts.copy()
            M, _ = ts_contaminated.shape

            if not explainer:  # use random series
                missing_rate = utils.verification_limitation(missing_rate)
                series_impacted = utils.verification_limitation(series_impacted)
                protection = utils.verification_limitation(protection)

                nbr_series_impacted = int(np.ceil(M * series_impacted))
                series_selected = [str(idx) for idx in np.random.choice(M, nbr_series_impacted, replace=False)]

            else:  # use fix series
                series_selected = [str(series_impacted)]

            if not explainer:
                print("\n\nMCAR contamination has been called with :"
                      "\n\ta number of series impacted ", series_impacted * 100, "%",
                      "\n\ta missing rate of ", missing_rate * 100, "%",
                      "\n\ta starting position at ", protection,
                      "\n\ta block size of ", block_size,
                      "\n\twith a seed option set to ", use_seed,
                      "\n\tshape of the set ", ts_contaminated.shape,
                      "\n\tthis selection of series", *series_selected, "\n\n")

            for series in series_selected:
                S = int(series)
                N = len(ts_contaminated[S])  # number of values in the series
                P = int(N * protection)  # values to protect in the beginning of the series
                W = int((N - P) * missing_rate)  # number of data to remove
                B = int(W / block_size)  # number of block to remove

                if B <= 0:
                    raise ValueError("The number of block to remove must be greater than 0. "
                                     "The dataset or the number of blocks may not be appropriate.")

                data_to_remove = np.random.choice(range(P, N), B, replace=False)

                for start_point in data_to_remove:
                    for jump in range(block_size):  # remove the block size for each random position
                        position = start_point + jump

                        if position >= N:  # If block exceeds the series length
                            position = P + (position - N)  # Wrap around to the start after protection

                        while np.isnan(ts_contaminated[S, position]):
                            position = position + 1

                            if position >= N:  # If block exceeds the series length
                                position = P + (position - N)  # Wrap around to the start after protection

                        ts_contaminated[S, position] = np.nan

            return ts_contaminated

        def missing_percentage(ts, series_impacted=0.2, missing_rate=0.2, protection=0.1):
            """
            Apply missing percentage contamination to the time series data.

            Parameters
            ----------
            ts : numpy.ndarray
                The time series dataset to contaminate.
            series_impacted : float, optional
                Percentage of series to contaminate (default is 0.2).
            missing_rate : float, optional
                Percentage of missing values per series (default is 0.2).
            protection : float, optional
                Size of the uncontaminated section at the beginning of the series (default is 0.1).

            Returns
            -------
            numpy.ndarray
                The contaminated time series data.
            """

            ts_contaminated = ts.copy()
            M, _ = ts_contaminated.shape

            missing_rate = utils.verification_limitation(missing_rate)
            series_impacted = utils.verification_limitation(series_impacted)
            protection = utils.verification_limitation(protection)

            nbr_series_impacted = int(np.ceil(M * series_impacted))

            print("\n\nMISSING PERCENTAGE contamination has been called with :"
                  "\n\ta number of series impacted ", series_impacted * 100, "%",
                  "\n\ta missing rate of ", missing_rate * 100, "%",
                  "\n\ta starting position at ", protection,
                  "\n\tshape of the set ", ts_contaminated.shape,
                  "\n\tthis selection of series 0 to ", nbr_series_impacted, "\n\n")

            for series in range(0, nbr_series_impacted):
                S = int(series)
                N = len(ts_contaminated[S])  # number of values in the series
                P = int(N * protection)  # values to protect in the beginning of the series
                W = int((N - P) * missing_rate)  # number of data to remove

                for to_remove in range(0, W):
                    index = P + to_remove
                    ts_contaminated[S, index] = np.nan

            return ts_contaminated

        def blackout(ts, missing_rate=0.2, protection=0.1):
            """
            Apply blackout contamination to the time series data.

            Parameters
            ----------
            ts : numpy.ndarray
                The time series dataset to contaminate.
            missing_rate : float, optional
                Percentage of missing values per series (default is 0.2).
            protection : float, optional
                Size of the uncontaminated section at the beginning of the series (default is 0.1).

            Returns
            -------
            numpy.ndarray
                The contaminated time series data.
            """
            return TimeSeries.Contaminate.missing_percentage(ts, series_impacted=1, missing_rate=missing_rate,
                                                             protection=protection)
