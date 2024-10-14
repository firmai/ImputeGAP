import os
import time
import numpy as np
import matplotlib
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

from imputegap.tools import utils

if os.getenv('CI') is None:
    matplotlib.use('TkAgg')

from matplotlib import pyplot as plt  # type: ignore


class TimeSeries:

    def __init__(self):
        """
        IMPORT FORMAT : (Values,Series) : series are seperated by "SPACE" et values by "\\n"
        """
        self.data = None

    def import_matrix(self, data=None):
        """
        Load timeseries manager from file
        FORMAT : (Series,Values), values are seperated by space et series by \n
        @author Quentin Nater

        :param data: matrix of time series
        :return: time series format for imputegap from dataset
        """
        if data is not None:
            if isinstance(data, list):
                self.data = np.array(data)

            elif isinstance(data, np.ndarray):
                self.data = data
            else:
                print("\nThe time series has not been loaded, format unknown\n")
                self.data = None

            return self

    def load_timeseries(self, data=None, max_series=None, max_values=None):
        """
        Load timeseries manager from file
        FORMAT : (Values,Series), values are seperated by space et series by \n
        @author Quentin Nater

        :param filename: path of the time series dataset
        :param max_series : limitation of the maximum number of series (computation limitation) | default None
        :param max_values : limitation of the maximum number of values by series (computation limitation) | default None
        :return: time series format for imputegap from dataset
        """

        if data is not None:
            if isinstance(data, str):
                print("\nThe time series has been loaded from " + str(data) + "\n")
                self.data = np.genfromtxt(data, delimiter=' ', max_rows=max_values)

                if max_series is not None:
                    self.data = self.data[:, :max_series]
            else:
                print("\nThe time series has not been loaded, format unknown\n")
                self.data = None

            self.data = self.data.T

            return self

    def print(self, limit=10, view_by_series=False):
        """
        Display the limited series from your dataset
        @author Quentin Nater

        :param limit: Number of series to print
        :param view_by_series: View (Values/Series) if true, (Series/Values) if false
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
        Display the result of the imputation
        :param metrics : [OPTIONAL], metrics to print in dictionary
        :param algorithm : [OPTIONAL], print the algorithm used
        @author Quentin Nater
        """
        print("\n\nResults of the imputation ", algorithm, " :")
        for key, value in metrics.items():
            print(f"{key:<20} = {value}")
        print("\n")

    def normalize(self, normalizer="z_score"):
        """
        Normalization of a dataset with "z_score", "min_max"
        @author Quentin Nater

        :param normalizer: ("z_score", "min_max"), normalization technic to use | default = "z_score"
        :return: data_normalized, normalized dataset
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
             max_values=None, size=(16, 8), save_path="", display=True):
        """
        Plot a chosen time series
        @author Quentin Nater

        :param raw_data: original time series without contamination
        :param infected_data: time series with contamination
        :param imputed_data: new time series with imputation values
        :param title: title of the plot
        :param max_series : limitation of the maximum number of series (computation limitation) | default None
        :param max_values : limitation of the maximum number of values by series (computation limitation) | default None
        :param size : size of the plots
        :param save_path : path to save locally the plot
        :param display : display or not the result

        :return : filepath
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

            for i in range(raw_data.shape[0]):
                color = colors[i % len(colors)]

                if infected_data is None and imputed_data is None:  # plot only raw matrix
                    plt.plot(np.arange(min(raw_data.shape[1], max_values)), raw_data[i, :max_values], linewidth=2.5,
                             color=color,
                             linestyle='-', label=f'TS {i + 1}')

                if infected_data is not None and imputed_data is None:  # plot infected matrix

                    if np.isnan(infected_data[i, :]).any():
                        plt.plot(np.arange(min(raw_data.shape[1], max_values)), raw_data[i, :max_values], linewidth=1.5, color='r', linestyle='--', label=f'TS-MB {i + 1}')

                    plt.plot(np.arange(min(infected_data.shape[1], max_values)), infected_data[i, :max_values], color=color, linewidth=2.5, linestyle='-', label=f'TS-RAW {i + 1}')



                if imputed_data is not None:  # plot imputed matrix

                    if np.isnan(infected_data[i, :]).any():
                        plt.plot(np.arange(min(imputed_data.shape[1], max_values)), imputed_data[i, :max_values], linestyle='-', color="r", label=f'TS-IMP {i + 1}')

                    if np.isnan(infected_data[i, :]).any():
                        plt.plot(np.arange(min(raw_data.shape[1], max_values)), raw_data[i, :max_values], linewidth=1.5, linestyle='--', color=color, label=f'TS-MB {i + 1}')

                    plt.plot(np.arange(min(infected_data.shape[1], max_values)), infected_data[i, :max_values], color=color, linewidth=2.5, linestyle='-', label=f'TS-RAW {i + 1}')



                number_of_series += 1
                if number_of_series == max_series:
                    break

        plt.xlabel('Time Shift')
        plt.ylabel('Values')
        plt.title(title)
        plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1.5)

        file_path = None
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path + "/" + title.replace(" ", "") + "_graph.png")
            plt.savefig(file_path, bbox_inches='tight')
            print("plots saved in ", file_path)

        if display:
            plt.show()

        #plt.close()

        return file_path


    class Contaminate:

        def mcar(ts, series_impacted=0.2, missing_rate=0.2, block_size=10, protection=0.1, use_seed=True, seed=42,
                 explainer=False):
            """
            Contamination of time series base on the Missing Completely at Random scenario

            :param series_impacted: percentage of series contaminated | default 0.2
            :param missing_rate: percentage of missing values by series  | default 0.2
            :param block_size: size of the block to remove at each random position selected  | default 10
            :param protection: size in the beginning of the time series where contamination is not proceeded  | default 0.1
            :param use_seed: use a seed to reproduce the test | default true
            :param seed: value of the seed | default 42
            :param explainer : use the MCAR on specific series to explain the imputation # default False
            :return: the contaminated time series
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
                series_indices = [str(idx) for idx in np.random.choice(M, nbr_series_impacted, replace=False)]

            else:  # use fix series
                series_indices = [str(series_impacted)]

            series_selected = utils.format_selection(ts_contaminated, series_indices)

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
            Contamination of time series base on the missing percentage scenario

            :param series_impacted: percentage of series contaminated | default 0.2
            :param missing_rate: percentage of missing values by series  | default 0.2
            :param protection: size in the beginning of the time series where contamination is not proceeded  | default 0.1
            :return: the contaminated time series
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
            Contamination of time series base on the blackout scenario

            :param missing_rate: percentage of missing values by series  | default 0.2
            :param protection: size in the beginning of the time series where contamination is not proceeded  | default 0.1
            :return: the contaminated time series
            """
            return TimeSeries.Contaminate.missing_percentage(ts, series_impacted=1, missing_rate=missing_rate,
                                                             protection=protection)
