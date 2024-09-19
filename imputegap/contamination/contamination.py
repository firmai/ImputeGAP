import math
import numpy as np


class Contamination:

    def format_selection(ts, selection):
        """
        Format the selection of series based on keywords
        @author Quentin Nater

        :param selection: current selection of series
        :param ts: dataset to contaminate
        :return series_selected : correct format of selection series
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

    def scenario_mcar(ts, series_impacted=0.1, missing_rate=0.1, block_size=10, protection=0.1, use_seed=True, seed=42):

        if use_seed:
            np.random.seed(seed)

        ts_contaminated = ts.copy()
        M, _ = ts_contaminated.shape

        nbr_series_impacted = int(np.ceil(M * series_impacted))
        series_indices = [str(idx) for idx in np.random.choice(M, nbr_series_impacted, replace=False)]
        series_selected = Contamination.format_selection(ts_contaminated, series_indices)


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
            P = int(N * protection)  # values to protect in the begining of the series
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
                        position = position+1

                        if position >= N:  # If block exceeds the series length
                            position = P + (position - N)  # Wrap around to the start after protection

                    ts_contaminated[S, position] = np.nan

        return ts_contaminated