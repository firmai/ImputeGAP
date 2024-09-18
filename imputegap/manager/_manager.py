import os
import numpy as np
from matplotlib import pyplot as plt  # type: ignore
from imputegap.contamination._contamination import ContaminationGAP
from imputegap.imputation._imputation import ImputationGAP


class TimeSeriesGAP:

    def __init__(self, filename):
        """
        :param filename: file path to the time series dataset
        :param ts : Original time series without alteration (ground-truth)
        :param contaminated_ts : time series after contamination
        :param imputation : time series after reconstruction of the missing data
        :param optimal_params : optimal parameters found for a specific algorithm and time series dataset
        :param explainer : result of the shap algorithm to explain the imputation of the time series dataset
        """
        self.filename = filename
        self.ts = self.load_timeseries()
        self.normalized_ts = None
        self.contaminated_ts = None
        self.imputation = None
        self.imputation_metrics = []
        self.optimal_params = None
        self.explainer = None

    def load_timeseries(self):
        """
        Load timeseries manager from file
        FORMAT : (Values,Series), values are seperated by space et series by \n
        @author Quentin Nater

        :param filename: path of the time series dataset
        :return: panda set of series transposed
        """

        print("\nThe time series has been loaded from " + str(self.filename) + "\n")
        time_series = np.genfromtxt(self.filename, delimiter=' ')
        self.ts = time_series.T

        return self.ts

    def print(self, limitation=10):
        """
        Display the limited series from your dataset
        @author Quentin Nater

        :param limitation: Number of series to print
        """

        print("\nGround-truth set :")
        for i, series in enumerate(self.ts[:limitation]):
            print(f"Series {i} " + " ".join([f"{elem:6}" for elem in series]))
        if limitation < self.ts.shape[0]:
            print("...")


        if self.normalized_ts is not None:
            print("\nGround-truth set normalized :")
            for i, series in enumerate(self.normalized_ts[:limitation]):
                print(f"Series {i} " + " ".join([f"{elem:6}" for elem in series]))
            if limitation < self.ts.shape[0]:
                print("...")

        if self.contaminated_ts is not None:
            print("\nContaminated set :")
            for i, series in enumerate(self.contaminated_ts[:limitation]):
                print(f"Series {i} " + " ".join([f"{elem:6}" for elem in series]))
            if limitation < self.contaminated_ts.shape[0]:
                print("...")

        if self.imputation is not None:
            print("\nImputation set :")
            for i, series in enumerate(self.imputation[:limitation]):
                print(f"Series {i} " + " ".join([f"{elem:6}" for elem in series]))
            if limitation < self.imputation.shape[0]:
                print("...")

        print("\nshape : ", self.ts.shape[0], " x ", self.ts.shape[1], "\n")

    def print_results(self):
        """
        Display the result of the imputation
        @author Quentin Nater
        """
        print("\n\nResults of the imputation : ")
        for key, value in self.imputation_metrics.items():
            print(f"{key:<20} = {value}")
        print("\n")

    def normalization_min_max(self):
        """
        Normalization of a dataset with MIN/MAX
        @author Quentin Nater

        :param ts: time series to normalize
        :return: data_normalized, normalized dataset
        """
        print("Normalization of the original time series dataset with min/max...")

        ts_min = self.ts.min(axis=0)  # Min for each series
        ts_max = self.ts.max(axis=0)  # Max for each series

        range_ts = ts_max - ts_min
        range_ts[range_ts == 0] = 1  # To avoid division by zero for constant series

        data_normalized = (self.ts - ts_min) / range_ts
        self.normalized_ts = data_normalized

        return self.normalized_ts

    def normalization_z_score(self):
        """
        Normalization of a dataset with Z-Score
        @author Quentin Nater

        :param ts: time series to normalize
        :return: data_normalized, normalized dataset
        """
        print("Normalization of the original time series dataset with Z-Score...")

        mean = np.mean(self.ts)
        std_dev = np.std(self.ts)

        z_scores = (self.ts - mean) / std_dev
        self.normalized_ts = z_scores

        return self.normalized_ts

    def plot(self, ts_type="ground_truth", title='Time Series Data', save_path="", limitation=10, size=(16, 8),
             display=True, colors=['dimgrey', 'plum', 'lightblue', 'mediumseagreen', 'khaki']):
        """
        Plot a chosen time series
        @author Quentin Nater

        :param title: title of the plot
        :param save_path : path to save locally the plot
        :param limitation: number of series displayed inside the plot
        :param size : size of the plots
        :param display : display or not the result
        :param colors : colors for each time series
        """
        number_of_series = 0
        plt.figure(figsize=size)

        if ts_type == "ground_truth":
            for i in range(self.ts.shape[0]):
                plt.plot(np.arange(self.ts.shape[1]), self.ts[i, :], label=f'Series {i + 1}')
                number_of_series += 1

                if number_of_series == limitation:
                    break

        elif ts_type == "ground_truth_normalized":
            for i in range(self.normalized_ts.shape[0]):
                plt.plot(np.arange(self.normalized_ts.shape[1]), self.normalized_ts[i, :], label=f'Series {i + 1}')
                number_of_series += 1

                if number_of_series == limitation:
                    break

        elif ts_type == "contamination":
            for i in range(self.ts.shape[0]):
                color = colors[i % len(colors)]

                plt.plot(np.arange(self.ts.shape[1]), self.ts[i, :], 'r--', label=f'Series {i + 1}-GT')
                plt.plot(np.arange(self.contaminated_ts.shape[1]), self.contaminated_ts[i, :], linewidth=2.5,
                         color=color, linestyle='-', label=f'Series {i + 1}-MV')

                number_of_series += 1
                if number_of_series == limitation:
                    break

        elif ts_type == "imputation":
            for i in range(self.ts.shape[0]):
                color = colors[i % len(colors)]
                plt.plot(np.arange(self.imputation.shape[1]), self.imputation[i, :], linewidth=2.5, color=color,
                         linestyle='-', label=f'Series{i + 1}-IMP')

                number_of_series += 1
                if number_of_series == limitation:
                    break

        plt.xlabel('Time Shift')
        plt.ylabel('Values')
        plt.title(title)
        plt.legend(loc='upper right', bbox_to_anchor=(1.10, 1))

        if save_path:
            file_path = os.path.join(save_path, title + "_" + ts_type + ".png")
            plt.savefig(file_path, bbox_inches='tight')
            print("plots saved in ", file_path)

        if display:
            plt.show()

        plt.close()

    def contamination_mcar(self, ts=None, missing_rate=0.1, block_size=10, series_selected=["*"],
                           starting_position=0.1, use_seed=True, seed=42):
        """
        Contamination with MCAR scenario
        @author Quentin Nater

        :param missing_rate: total percentage of contamination
        :param block_size: size of the contamination from a random point
        :param series_selected: series to contaminate
        :param starting_position : all elements before this position is protected from contamination
        :param use_seed : use seed value as random constant to reproduce the experimentation
        :param seed : seed value for random constant
        :return: all time series with and without contamination
        """
        if ts is None:
            ts = self.ts

        self.contaminated_ts = ContaminationGAP(self).contamination_mcar(ts, missing_rate, block_size,
                                                                         series_selected, starting_position, use_seed,
                                                                         seed)

    def imputation_cdrec(self, ground_truth=None, contamination=None, params=None):
        """
        Imputation of data with CDREC algorithm
        @author Quentin Nater

        :param ground_truth: original time series without contamination, if None, self are loaded
        :param contamination: time series with contamination, if None, self are loaded
        :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

        :return: all time series with imputation data
        """
        if ground_truth is None:
            ground_truth = self.ts
        if contamination is None:
            contamination = self.contaminated_ts

        self.imputation, self.imputation_metrics = ImputationGAP().cdrec(ground_truth, contamination, params)

        return self.imputation, self.imputation_metrics
