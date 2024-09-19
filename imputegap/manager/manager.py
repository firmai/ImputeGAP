import os
import numpy as np
from matplotlib import pyplot as plt  # type: ignore
from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation


class TimeSeries:

    def __init__(self, filename, normalization=None):
        """
        :param ts : Original time series without alteration (ground-truth)
        :param contaminated_ts : time series after contamination
        :param imputation : time series after reconstruction of the missing data
        :param optimal_params : optimal parameters found for a specific algorithm and time series dataset
        :param explainer : result of the shap algorithm to explain the imputation of the time series dataset
        """
        self.ts = self.load_timeseries(filename, normalization)
        self.ts_contaminate = None
        self.ts_imputation = None
        self.metrics = []
        self.optimal_params = None
        self.explainer = None

    def load_timeseries(self, filename, normalization=None):
        """
        Load timeseries manager from file
        FORMAT : (Values,Series), values are seperated by space et series by \n
        @author Quentin Nater

        :param filename: path of the time series dataset
        :param normalization : [OPTIONAL] choice of normalization ("z_score" or "min_max")
        :return: time series format for imputegap from dataset
        """

        print("\nThe time series has been loaded from " + str(filename) + "\n")

        time_series = np.genfromtxt(filename, delimiter=' ')
        ts = time_series.T

        if normalization is not None:
            if normalization == "z_score":
                ts = self.normalization_z_score(ts)
            elif normalization == "min_max":
                ts = self.normalization_min_max(ts)
            else:
                print("Normalization asked is not registered...\n")

        return ts

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

        if self.ts_contaminate is not None:
            print("\nContaminated set :")
            for i, series in enumerate(self.ts_contaminate[:limitation]):
                print(f"Series {i} " + " ".join([f"{elem:6}" for elem in series]))
            if limitation < self.ts_contaminate.shape[0]:
                print("...")

        if self.ts_imputation is not None:
            print("\nImputation set :")
            for i, series in enumerate(self.ts_imputation[:limitation]):
                print(f"Series {i} " + " ".join([f"{elem:6}" for elem in series]))
            if limitation < self.ts_imputation.shape[0]:
                print("...")

        print("\nshape : ", self.ts.shape[0], " x ", self.ts.shape[1], "\n")

    def print_results(self):
        """
        Display the result of the imputation
        @author Quentin Nater
        """
        print("\n\nResults of the imputation : ")
        for key, value in self.metrics.items():
            print(f"{key:<20} = {value}")
        print("\n")

    def normalization_min_max(self, ts):
        """
        Normalization of a dataset with MIN/MAX
        @author Quentin Nater

        :param ts: time series to normalize
        :return: data_normalized, normalized dataset
        """
        print("Normalization of the original time series dataset with min/max...")

        ts_min = ts.min(axis=0)  # Min for each series
        ts_max = ts.max(axis=0)  # Max for each series

        range_ts = ts_max - ts_min
        range_ts[range_ts == 0] = 1  # To avoid division by zero for constant series

        min_max = (ts - ts_min) / range_ts

        return min_max

    def normalization_z_score(self, ts):
        """
        Normalization of a dataset with Z-Score
        @author Quentin Nater

        :param ts: time series to normalize
        :return: data_normalized, normalized dataset
        """
        print("Normalization of the original time series dataset with Z-Score...")

        mean = np.mean(ts)
        std_dev = np.std(ts)

        z_scores = (ts - mean) / std_dev

        return z_scores

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
                plt.plot(np.arange(self.ts_contaminate.shape[1]), self.ts_contaminate[i, :], linewidth=2.5,
                         color=color, linestyle='-', label=f'Series {i + 1}-MV')

                number_of_series += 1
                if number_of_series == limitation:
                    break

        elif ts_type == "imputation":
            for i in range(self.ts.shape[0]):
                color = colors[i % len(colors)]
                plt.plot(np.arange(self.ts_imputation.shape[1]), self.ts_imputation[i, :], linewidth=2.5, color=color,
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