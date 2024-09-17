import math
import numpy as np


class ContaminationGAP:

    def __init__(self, gap=None):
        """
        Initialize the ContaminationGAP class.

        :param manager: the TimeSeriesGAP object managing the time series.
        """
        self.gap = gap

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
            for i in range(0, self.ts.shape[0] - ending):
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

    def contamination_mcar(self, missing_rate=0.1, block_size=10, starting_position=0.1, series_selected=["*"],
                           use_seed=True, seed=42):
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

        ts_contaminated = self.gap.ts.copy()
        n_series, n_values = ts_contaminated.shape
        series_selected = self.format_selection(series_selected)

        # protect the % before the contamination
        start_index = int(math.ceil((n_values * starting_position)))

        population = (n_values - start_index) * len(series_selected)

        to_remove = int(math.ceil(population * missing_rate))

        block_to_remove = int(to_remove / block_size)

        print("\nMCAR contamination has been called with :"
              "\n\ta missing rate of ", missing_rate * 100, "%",
              "\n\ta starting position at ", start_index,
              "\n\ta block size of ", block_size,
              "\n\twith a seed option set to ", use_seed,
              "\n\tshape of the set ", ts_contaminated.shape,
              "\n\tthis selection of series", *series_selected,
              "\n\tfor a total population of ", population,
              "\n\tnumber of values to removed set to ", to_remove,
              "\n\tblocks to remove ", block_to_remove, "\n")

        if use_seed:
            np.random.seed(seed)

        if block_to_remove <= 0:
            raise ValueError("The number of block to remove must be greater than 0. "
                             "The dataset or the number of blocks may not be appropriate.")

        missing_indices = np.random.choice(population, int(to_remove / block_size), replace=False)

        for index in missing_indices:
            for current_block_jump in range(0, block_size):
                row = int(series_selected[index % len(series_selected)])
                col = (index // len(series_selected)) + start_index + current_block_jump
                if col >= n_values:  # outbound limitation
                    col = col - n_values + start_index
                ts_contaminated[row, col] = np.nan

        self.gap.contaminated_ts = ts_contaminated

        return self.gap.contaminated_ts
