import math
import os

import numpy as np

from matplotlib import pyplot as plt # type: ignore

class TimeSeriesGAP:

    def __init__(self, filename):
        self.filename = filename
        self.ts = self.load_timeseries()
        self.contaminated_ts = None

    def load_timeseries (self) :
        """
        Load timeseries manager from file
        FORMAT : (Values,Series), values are seperated by space et series by \n
        @author Quentin Nater

        :param filename: path of the time series dataset
        :return: panda set of series transposed
        """

        print("\nThe time series has been loaded from " + str(self.filename)+ "\n")
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

        print("\nContaminated set :")
        if self.contaminated_ts is not None:
            for i, series in enumerate(self.contaminated_ts[:limitation]):
                print(f"Series {i} " + " ".join([f"{elem:6}" for elem in series]))
            if limitation < self.contaminated_ts.shape[0]:
                print("...")

        print("\nshape : ", self.ts.shape[0], " x ", self.ts.shape[1], "\n")


    def plot(self, ts_type="ground_truth", title='Time Series Data', save_path="", limitation=10, size=(16,8), display=True, colors = ['dimgrey', 'plum', 'lightblue', 'mediumseegreen', 'khaki']):
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

        elif ts_type == "contamination":
            for i in range(self.ts.shape[0]):
                color = colors[i % len(colors)]

                plt.plot(np.arange(self.ts.shape[1]), self.ts[i, :], 'r--',label=f'Series {i + 1}-GT')
                plt.plot(np.arange(self.contaminated_ts.shape[1]), self.contaminated_ts[i, :], linewidth=2.5, color=color, linestyle='-', label=f'Series {i + 1}-MV')

                number_of_series += 1
                if number_of_series == limitation:
                    break

        plt.xlabel('Time Shift')
        plt.ylabel('Values')
        plt.title(title)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

        if save_path:
            file_path = os.path.join(save_path, title + "_" + ts_type + ".png")
            plt.savefig(file_path)
            print("plots saved in ", file_path)

        if display:
            plt.show()

    def format_selection(self, selection):
        """
        Format the selection of series based on keywords
        @author Quentin Nater

        :param selection: current selection of series
        :return series_selected : correct format of selection series
        """
        if not selection:
            selection = ["*"]

        if selection == ["*"]:
            series_selected = []
            for i in range(0, self.ts.shape[0]):
                series_selected.append(str(i))
            return series_selected

        elif "-" in selection[0]:
            series_selected = []
            value = selection[0]
            ending = int(value[1:])
            for i in range(0, self.ts.shape[0]-ending):
                series_selected.append(str(i))
            return series_selected

        elif "+" in selection[0]:
            series_selected = []
            value = selection[0]
            starting = int(value[1:])
            for i in range(starting, self.ts.shape[0]):
                series_selected.append(str(i))
            return series_selected

        else:
            return selection


    def contamination_mcar(self, missing_rate=0.1, block_size=10, starting_position=0.1, series_selected=["*"], use_seed=True, seed=42):
        """
        Contamination with MCAR scenario
        @author Quentin Nater

        :param ts: time series to contaminate
        :param missing_rate: total percentage of contamination
        :param block_size: size of the contamination from a random point
        :param starting_position : all elements before this position is protected from contamination
        :param series_selected: series to contaminate
        :param use_seed : use seed value as random constant to reproduce the experimentation
        :param seed : seed value for random constant
        :return: all time series with and without contamination
        """

        ts_contaminated = self.ts.copy()
        n_series, n_values = ts_contaminated.shape
        series_selected = self.format_selection(series_selected)

        # protect the % before the contamination
        start_index = int(math.ceil((n_values * starting_position)))

        population = (n_values - start_index) * len(series_selected)

        to_remove = int(math.ceil(population * missing_rate))

        block_to_remove = int(to_remove/block_size)

        print("\nMCAR contamination has been called with :"
              "\n\ta missing rate of ", missing_rate*100, "%",
              "\n\ta starting position at ", start_index,
              "\n\ta block size of ", block_size,
              "\n\twith a seed option set to ", use_seed,
              "\n\tshape of the set ",  ts_contaminated.shape,
              "\n\tthis selection of series", *series_selected,
              "\n\tfor a total population of ", population,
              "\n\tnumber of values to removed set to ", to_remove,
              "\n\tblocks to remove ", block_to_remove, "\n")

        if use_seed:
            np.random.seed(seed)

        if block_to_remove <= 0:
            raise ValueError("The number of block to remove must be greater than 0. "
                             "The dataset or the number of blocks may not be appropriate.")

        missing_indices = np.random.choice(population, int(to_remove/block_size), replace=False)

        for index in missing_indices:
            for current_block_jump in range(0, block_size):
                row = int(series_selected[index % len(series_selected)])
                col = (index // len(series_selected)) + start_index + current_block_jump
                if col >= n_values: # outbound limitation
                    col = col - n_values + start_index
                ts_contaminated[row, col] = np.nan

        self.contaminated_ts = ts_contaminated

        return self.contaminated_ts
