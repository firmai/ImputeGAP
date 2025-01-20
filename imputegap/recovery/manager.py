import datetime
import os
import time
import numpy as np
import matplotlib
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import importlib.resources
from scipy.stats import norm

from imputegap.tools import utils

# Use Agg backend if in a headless or CI environment
if os.getenv('DISPLAY') is None or os.getenv('CI') is not None:
    matplotlib.use("Agg")
    print("Running in a headless environment or CI. Using Agg backend.")
else:
    try:
        matplotlib.use("TkAgg")
        if importlib.util.find_spec("tkinter") is None:
            print("tkinter is not available.")
    except (ImportError, RuntimeError):
        matplotlib.use("Agg")

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

    load_series(data=None, max_series=None, max_values=None, header=False) :
        Loads time series data from a file or predefined dataset.

    print(limit=10, view_by_series=False) :
        Prints a limited number of time series from the dataset.

    print_results(metrics, algorithm="") :
        Prints the results of the imputation process.

    normalize(normalizer="z_score") :
        Normalizes the time series dataset.

    plot(input_data, incomp_data=None, recov_data=None, max_series=None, max_values=None, size=(16, 8), save_path="", display=True) :
        Plots the time series data, including raw, contaminated, or imputed data.

    Contamination :
        Class containing methods to contaminate time series data with missing values based on different patterns.

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
        """
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
                print("\nThe time series have not been loaded, format unknown\n")
                self.data = None
                raise ValueError("Invalid input for import_matrix")

            return self

    def load_series(self, data, max_series=None, max_values=None, header=False):
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
                saved_data = data

                #  update path form inner library datasets
                if data in ["bafu.txt", "chlorine.txt", "climate.txt", "drift.txt", "eeg-alcohol.txt",
                            "eeg-reading.txt",
                            "meteo.txt", "test.txt", "test-large.txt", "fmri-objectviewing.txt", "fmri-stoptask.txt"]:
                    data = importlib.resources.files('imputegap.dataset').joinpath(data)

                if not os.path.exists(data):
                    data = ".." + saved_data
                    if not os.path.exists(data):
                        data = data[1:]

                self.data = np.genfromtxt(data, delimiter=' ', max_rows=max_values, skip_header=int(header))

                print("\nThe time series have been loaded from " + str(data) + "\n")

                if max_series is not None:
                    self.data = self.data[:, :max_series]
            else:
                print("\nThe time series have not been loaded, format unknown\n")
                self.data = None
                raise ValueError("Invalid input for load_series")

            self.data = self.data.T

            return self

    def print(self, limit_timestamps=10, limit_series=7, view_by_series=False):
        """
        Prints a limited number of time series from the dataset.

        Parameters
        ----------
        limit_timestamps : int, optional
        The number of timestamps to print. Default is 15. Use -1 for no restriction.
        limit_series : int, optional
        The number of series to print. Default is 10. Use -1 for no restriction.
        view_by_series : bool, optional
        Whether to view by series (True) or by values (False).

        Returns
        -------
        None
        """
        print("\nTime Series set :")

        to_print = self.data
        nbr_series, nbr_values = to_print.shape
        print_col, print_row = "Timestamp", "Series"

        if limit_timestamps == -1:
            limit_timestamps = to_print.shape[1]
        if limit_series == -1:
            limit_series = to_print.shape[0]
        to_print = to_print[:limit_series, :limit_timestamps]

        if not view_by_series:
            to_print = to_print.T
            print_col, print_row = "Series", "Timestamp"

        header_format = "{:<15}"  # Fixed size for headers
        value_format = "{:>15.10f}"  # Fixed size for values
        # Print the header
        print(f"{'':<18}", end="")  # Empty space for the row labels
        for i in range(to_print.shape[1]):
            print(header_format.format(f"{print_col}_{i + 1}"), end="")
        print()

        # Print each limited series with fixed size
        for i, series in enumerate(to_print):
            print(header_format.format(f"{print_row} {i + 1}"), end="")
            print("".join([value_format.format(elem) for elem in series]))

        if limit_series < nbr_series:
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

        print("\n\nImputation Results of", algorithm, ":")
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

    def plot(self, input_data, incomp_data=None, recov_data=None, max_series=None, max_values=None, series_range=None,
             subplot=False, size=(16, 8), save_path="", display=True):
        """
        Plot the time series data, including raw, contaminated, or imputed data.

        Parameters
        ----------
        input_data : numpy.ndarray
            The original time series data without contamination.
        incomp_data : numpy.ndarray, optional
            The contaminated time series data.
        recov_data : numpy.ndarray, optional
            The imputed time series data.
        max_series : int, optional
            The maximum number of series to plot.
        max_values : int, optional
            The maximum number of values per series to plot.
        series_range : int, optional
            The index of a specific series to plot. If set, only this series will be plotted.
        subplot : bool, optional
            Print one time series by subplot or all in the same plot.
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

        if max_series is None or max_series == -1:
            max_series = input_data.shape[0]
        if max_values is None or max_values == -1:
            max_values = input_data.shape[1]

        if subplot:
            series_indices = [i for i in range(incomp_data.shape[0]) if np.isnan(incomp_data[i]).any()]
            nbr_series = [series_range] if series_range is not None else range(min(len(series_indices), max_series))
            n_series_to_plot = len(nbr_series)
        else:
            series_indices = [series_range] if series_range is not None else range(min(input_data.shape[0], max_series))
            n_series_to_plot = len(series_indices)

        if subplot:
            n_cols = min(3, n_series_to_plot)
            n_rows = (n_series_to_plot + n_cols - 1) // n_cols

            x_size, y_size = size
            x_size = x_size * n_cols
            y_size = y_size * n_rows

            scale_factor = 0.85
            x_size_screen = (1920 / 100) * scale_factor
            y_size_screen = (1080 / 100) * scale_factor

            if n_rows < 4:
                x_size = x_size_screen
                y_size = y_size_screen

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(x_size, y_size), squeeze=False)
            axes = axes.flatten()
        else:
            plt.figure(figsize=size)
            plt.grid(True, linestyle='--', color='#d3d3d3', linewidth=0.6)

        if input_data is not None:
            colors = utils.load_parameters("default", algorithm="colors")

            for idx, i in enumerate(series_indices):

                if subplot:
                    color = colors[0]
                else:
                    color = colors[i % len(colors)]

                timestamps = np.arange(min(input_data.shape[1], max_values))

                # Select the current axes if using subplots
                if subplot:
                    ax = axes[idx]
                    ax.grid(True, linestyle='--', color='#d3d3d3', linewidth=0.6)
                else:
                    ax = plt

                if incomp_data is None and recov_data is None:  # plot only raw matrix
                    ax.plot(timestamps, input_data[i, :max_values], linewidth=2.5,
                            color=color, linestyle='-', label=f'TS {i + 1}')

                if incomp_data is not None and recov_data is None:  # plot infected matrix
                    if np.isnan(incomp_data[i, :]).any():
                        ax.plot(timestamps, input_data[i, :max_values], linewidth=1.5,
                                color='r', linestyle='--', label=f'TS-INCOMP {i + 1}')

                    if np.isnan(incomp_data[i, :]).any() or not subplot:
                        ax.plot(np.arange(min(incomp_data.shape[1], max_values)), incomp_data[i, :max_values],
                            color=color, linewidth=2.5, linestyle='-', label=f'TS-INPUT {i + 1}')

                if recov_data is not None:  # plot imputed matrix
                    if np.isnan(incomp_data[i, :]).any():
                        ax.plot(np.arange(min(recov_data.shape[1], max_values)), recov_data[i, :max_values],
                                linestyle='-', color="r", label=f'TS-RECOV {i + 1}')

                        ax.plot(timestamps, input_data[i, :max_values], linewidth=1.5,
                                linestyle='--', color=color, label=f'TS-INCOM {i + 1}')

                    if np.isnan(incomp_data[i, :]).any() or not subplot:
                        ax.plot(np.arange(min(incomp_data.shape[1], max_values)), incomp_data[i, :max_values],
                                color=color, linewidth=2.5, linestyle='-', label=f'TS-INPUT {i + 1}')

                # Label and legend for subplot
                if subplot:
                    ax.set_xlabel('Timestamp', fontsize=7)
                    ax.set_ylabel('Values', fontsize=7)
                    ax.legend(loc='upper left', fontsize=7)
                    plt.tight_layout()

                number_of_series += 1
                if number_of_series == max_series:
                    break

        if subplot:
            for idx in range(len(series_indices), len(axes)):
                axes[idx].axis('off')

        if not subplot:
            plt.xlabel('Timestamp')
            plt.ylabel('Values')
            plt.legend(
                loc='upper left',
                fontsize=10,
                frameon=True,
                fancybox=True,
                shadow=True,
                borderpad=1.5,
                bbox_to_anchor=(1.02, 1),  # Adjusted to keep the legend inside the window
            )

        file_path = None

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            now = datetime.datetime.now()
            current_time = now.strftime("%y_%m_%d_%H_%M_%S")
            file_path = os.path.join(save_path + "/" + current_time + "_plot.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            print("plots saved in ", file_path)

        if display:
            plt.show()

        return file_path

    class Contamination:
        """
        Inner class to apply contamination patterns to the time series data.

        Methods
        -------
        mcar(ts, series_rate=0.2, missing_rate=0.2, block_size=10, offset=0.1, seed=True, explainer=False) :
            Apply Missing Completely at Random (MCAR) contamination to the time series data.

        missing_percentage(ts, series_rate=0.2, missing_rate=0.2, offset=0.1) :
            Apply missing percentage contamination to the time series data.

        blackout(ts, missing_rate=0.2, offset=0.1) :
            Apply blackout contamination to the time series data.

        gaussian(input_data, series_rate=0.2, missing_rate=0.2, std_dev=0.2, offset=0.1, seed=True):
            Apply Gaussian contamination to the time series data.

        def disjoint(input_data, missing_rate=0.1, limit=1, offset=0.1):
            Apply Disjoint contamination to the time series data.

        def overlap(input_data, missing_rate=0.2, limit=1, shift=0.05, offset=0.1,):
            Apply Overlapping contamination to the time series data.
        """

        def mcar(input_data, dataset_rate=0.2, series_rate=0.2, block_size=10, offset=0.1, seed=True, explainer=False):
            """
            Apply Missing Completely at Random (MCAR) contamination to the time series data.

            Parameters
            ----------
            input_data : numpy.ndarray
                The time series dataset to contaminate.
            dataset_rate : float, optional
                Percentage of series to contaminate (default is 0.2).
            series_rate : float, optional
                Percentage of missing values per series (default is 0.2).
            block_size : int, optional
                Size of the block of missing data (default is 10).
            offset : float, optional
                Size of the uncontaminated section at the beginning of the series (default is 0.1).
            seed : bool, optional
                Whether to use a seed for reproducibility (default is True).
            explainer : bool, optional
                Whether to apply MCAR to specific series for explanation purposes (default is False).

            Returns
            -------
            numpy.ndarray
                The contaminated time series data.
            """

            if seed:
                seed_value = 42
                np.random.seed(seed_value)

            ts_contaminated = input_data.copy()
            M, _ = ts_contaminated.shape

            if not explainer:  # use random series
                series_rate = utils.verification_limitation(series_rate)
                dataset_rate = utils.verification_limitation(dataset_rate)
                offset = utils.verification_limitation(offset)

                nbr_series_impacted = int(np.ceil(M * dataset_rate))
                series_selected = [str(idx) for idx in np.random.choice(M, nbr_series_impacted, replace=False)]

            else:  # use fix series
                series_selected = [str(dataset_rate)]

            if not explainer:
                print("\n\nMCAR contamination has been called with :"
                      "\n\ta number of series impacted ", dataset_rate * 100, "%",
                      "\n\ta missing rate of ", series_rate * 100, "%",
                      "\n\ta starting position at ", offset,
                      "\n\ta block size of ", block_size,
                      "\n\twith a seed option set to ", seed,
                      "\n\tshape of the set ", ts_contaminated.shape,
                      "\n\tthis selection of series", *series_selected, "\n\n")

            for series in series_selected:
                S = int(series)
                N = len(ts_contaminated[S])  # number of values in the series
                P = int(N * offset)  # values to protect in the beginning of the series
                W = int((N - P) * series_rate)  # number of data to remove
                B = int(W / block_size)  # number of block to remove

                if B <= 0:
                    raise ValueError("The number of block to remove must be greater than 0. "
                                     "The dataset or the number of blocks may not be appropriate."
                                     "One series has", str(N), "population is ", str((N - P)), "the number to remove",
                                     str(W), "and block site", str(block_size), "")

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

        def missing_percentage(input_data, dataset_rate=0.2, series_rate=0.2, offset=0.1):
            """
            Apply missing percentage contamination to the time series data.

            Parameters
            ----------
            input_data : numpy.ndarray
                The time series dataset to contaminate.
            dataset_rate : float, optional
                Percentage of series to contaminate (default is 0.2).
            series_rate : float, optional
                Percentage of missing values per series (default is 0.2).
            offset : float, optional
                Size of the uncontaminated section at the beginning of the series (default is 0.1).

            Returns
            -------
            numpy.ndarray
                The contaminated time series data.
            """

            ts_contaminated = input_data.copy()
            M, _ = ts_contaminated.shape

            series_rate = utils.verification_limitation(series_rate)
            dataset_rate = utils.verification_limitation(dataset_rate)
            offset = utils.verification_limitation(offset)

            nbr_series_impacted = int(np.ceil(M * dataset_rate))

            print("\n\nMISSING PERCENTAGE contamination has been called with :"
                  "\n\ta number of series impacted ", dataset_rate * 100, "%",
                  "\n\ta missing rate of ", series_rate * 100, "%",
                  "\n\ta starting position at ", offset,
                  "\n\tshape of the set ", ts_contaminated.shape,
                  "\n\tthis selection of series 0 to ", nbr_series_impacted, "\n\n")

            for series in range(0, nbr_series_impacted):
                S = int(series)
                N = len(ts_contaminated[S])  # number of values in the series
                P = int(N * offset)  # values to protect in the beginning of the series
                W = int((N - P) * series_rate)  # number of data to remove

                for to_remove in range(0, W):
                    index = P + to_remove
                    ts_contaminated[S, index] = np.nan

            return ts_contaminated

        def blackout(input_data, series_rate=0.2, offset=0.1):
            """
            Apply blackout contamination to the time series data.

            Parameters
            ----------
            input_data : numpy.ndarray
                The time series dataset to contaminate.
            series_rate : float, optional
                Percentage of missing values per series (default is 0.2).
            offset : float, optional
                Size of the uncontaminated section at the beginning of the series (default is 0.1).

            Returns
            -------
            numpy.ndarray
                The contaminated time series data.
            """
            return TimeSeries.Contamination.missing_percentage(input_data, dataset_rate=1, series_rate=series_rate, offset=offset)

        def gaussian(input_data, dataset_rate=0.2, series_rate=0.2, std_dev=0.2, offset=0.1, seed=True):
            """
            Apply contamination with a Gaussian distribution to the time series data.

            Parameters
            ----------
            input_data : numpy.ndarray
                The time series dataset to contaminate.
            dataset_rate : float, optional
                Percentage of series to contaminate (default is 0.2).
            series_rate : float, optional
                Percentage of missing values per series (default is 0.2).
            std_dev : float, optional
                Standard deviation of the Gaussian distribution for missing values (default is 0.2).
            offset : float, optional
                Size of the uncontaminated section at the beginning of the series (default is 0.1).
            seed : bool, optional
                Whether to use a seed for reproducibility (default is True).

            Returns
            -------
            numpy.ndarray
                The contaminated time series data.
            """

            ts_contaminated = input_data.copy()
            M, _ = ts_contaminated.shape

            if seed:
                seed_value = 42
                np.random.seed(seed_value)

            # Validation and limitation of input parameters
            series_rate = utils.verification_limitation(series_rate)
            dataset_rate = utils.verification_limitation(dataset_rate)
            offset = utils.verification_limitation(offset)

            nbr_series_impacted = int(np.ceil(M * dataset_rate))

            print("\n\nGAUSSIAN contamination has been called with :"
                  "\n\ta number of series impacted ", dataset_rate * 100, "%",
                  "\n\ta missing rate of ", series_rate * 100, "%",
                  "\n\ta starting position at ", offset,
                  "\n\tGaussian std_dev ", std_dev,
                  "\n\tshape of the set ", ts_contaminated.shape,
                  "\n\tthis selection of series 0 to ", nbr_series_impacted, "\n\n")

            for series in range(0, nbr_series_impacted):
                S = int(series)
                N = len(ts_contaminated[S])  # number of values in the series
                P = int(N * offset)  # values to protect in the beginning of the series
                W = int((N - P) * series_rate)  # number of data points to remove
                R = np.arange(P, N)

                # probability density function
                mean = np.mean(ts_contaminated[S])
                mean = max(min(mean, 1), -1)

                probabilities = norm.pdf(R, loc=P + mean * (N - P), scale=std_dev * (N - P))

                # normalizes the probabilities so that their sum equals 1
                probabilities /= probabilities.sum()

                # select the values based on the probability
                missing_indices = np.random.choice(R, size=W, replace=False, p=probabilities)

                # apply missing values
                ts_contaminated[S, missing_indices] = np.nan

            return ts_contaminated

        def disjoint(input_data, series_rate=0.1, limit=1, offset=0.1):
            """
            Apply disjoint contamination to the time series data.

            Parameters
            ----------
            input_data : numpy.ndarray
                The time series dataset to contaminate.
            series_rate : float, optional
                Percentage of missing values per series (default is 0.1).
            limit : float, optional
                Percentage expressing the limit index of the end of the contamination (default is 1: all length).
            offset : float, optional
                Size of the uncontaminated section at the beginning of the series (default is 0.1).

            Returns
            -------
            numpy.ndarray
                The contaminated time series data.
            """
            ts_contaminated = input_data.copy()
            M, N = ts_contaminated.shape

            series_rate = utils.verification_limitation(series_rate)
            offset = utils.verification_limitation(offset)

            print("\n\nDISJOINT contamination has been called with :"
                  "\n\ta missing rate of ", series_rate * 100, "%",
                  "\n\ta starting position at ", offset,
                  "\n\tshape of the set ", ts_contaminated.shape, "\n\n")

            S = 0
            X = int(len(ts_contaminated[0]) * offset) # current index with offset
            final_limit = int(N*limit)-1

            while S < M:
                N = len(ts_contaminated[S])  # number of values in the series
                P = int(N * offset)  # values to protect in the beginning of the series
                W = int((N - P) * series_rate)  # number of data to remove
                L = X + W  # new limit

                for to_remove in range(X, L):
                    index = P + to_remove
                    ts_contaminated[S, index] = np.nan

                    if index >= final_limit:  # reach the limitation
                        return ts_contaminated

                X = L + 1
                S = S + 1

            return ts_contaminated

        def overlap(input_data, series_rate=0.2, limit=1, shift=0.05, offset=0.1, ):
            """
            Apply overlap contamination to the time series data.

            Parameters
            ----------
            input_data : numpy.ndarray
                The time series dataset to contaminate.
            series_rate : float, optional
                Percentage of missing values per series (default is 0.2).
            limit : float, optional
                Percentage expressing the limit index of the end of the contamination (default is 1: all length).
            shift : float, optional
                Percentage of shift inside each the last disjoint contamination.
            offset : float, optional
                Size of the uncontaminated section at the beginning of the series (default is 0.1).

            Returns
            -------
            numpy.ndarray
                The contaminated time series data.
            """
            ts_contaminated = input_data.copy()
            M, N = ts_contaminated.shape

            series_rate = utils.verification_limitation(series_rate)
            offset = utils.verification_limitation(offset)

            print("\n\nOVERLAP contamination has been called with :"
                  "\n\ta missing rate of ", series_rate * 100, "%",
                  "\n\ta starting position at ", offset,
                  "\n\ta shift overlap of ", shift * 100, "%",
                  "\n\tshape of the set ", ts_contaminated.shape, "\n\n")

            S = 0
            X = int(len(ts_contaminated[0]) * offset)  # current index with offset
            X = X + int(X * shift)  # counter first shift
            final_limit = int(N * limit) - 1

            while S < M:
                N = len(ts_contaminated[S])  # number of values in the series
                P = int(N * offset)  # values to protect in the beginning of the series
                W = int((N - P) * series_rate)  # number of data to remove
                X = X - int(X * shift)
                L = X + W  # new limit

                for to_remove in range(X, L):
                    index = P + to_remove
                    ts_contaminated[S, index] = np.nan

                    if index >= final_limit:  # reach the limitation
                        return ts_contaminated

                X = L + 1
                S = S + 1

            return ts_contaminated
