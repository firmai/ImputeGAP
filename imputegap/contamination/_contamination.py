import math
import numpy as np


class TimeSeriesContamination:
    FACTOR_SHIFT = 0.1
    BLOCK_SIZE = 10

    def __init__(self, ts):
        """
        Initialize the class with the time series.

        :param ts: time series to be contaminated.
        """
        self.ts = ts


    def introduce_mcar(ts, missing_rate, series_selected, keep_other=False):
        """
        Contamination with MCAR scenario
        @author Quentin Nater

        :param ts: time series to contaminate
        :param missing_rate: percentage of contamination
        :param series_selected: series to contaminate
        :param keep_other: keep all series or only the contaminated one
        :return: the contaminated time series
        """

        print("\t\t\t>> MCAR : RATE ", missing_rate, " / keep other ", keep_other, " / series selected ", *series_selected, "*****\n")

        ts_contaminated = ts.copy()
        n_series, n_values = ts_contaminated.shape

        # protect the 10% before
        start_index = int(math.ceil((n_values * FACTOR_S)))

        if keep_other:
            population = (n_values - start_index) * len(series_selected)
        else:
            population = (n_values - start_index) * n_series

        to_remove = int(math.ceil(population * missing_rate))

        print("\t\t\t>> MCAR : population ", population, " / to_remove ", to_remove, "*****\n")

        np.random.seed(SEED)
        missing_indices = np.random.choice(population, int(to_remove/BLOCK_SIZE), replace=False)

        if keep_other:
            for index in missing_indices:
                for i in range(0, BLOCK_SIZE):
                    row = series_selected[index % len(series_selected)]
                    col = (index // len(series_selected)) + start_index

                    if col >= (n_values - BLOCK_SIZE - 1):
                        col = col - (n_values - start_index)

                    ts_contaminated.iat[row, col+i] = np.nan
        else:
            for index in missing_indices:
                for i in range(0, BLOCK_SIZE):
                    row = index % n_series
                    col = (index // n_series) + start_index + i
                    if col >= (n_values - start_index):
                        col = col - (n_values - start_index)
                    ts_contaminated.iat[row, col] = np.nan

        return ts_contaminated, ts_contaminated.to_numpy()



    def introduce_missingpourcentage(self, missing_rate, series_selected, keep_other=False):
        """
        Contamination with Missing-Pourcentage scenario
        @author Quentin Nater

        :param ts: time series to contaminate
        :param missing_rate: percentage of contamination
        :param series_selected: series to contaminate
        :param keep_other: keep all series or only the contaminated one
        :return: the contaminated time series
        """

        print("\t\t\t>> MISSING POURCENTAGE : RATE ", missing_rate, " / keep other ", keep_other, " / series selected ", *series_selected, "*****\n")

        ts_contaminated = self.ts.copy()
        n_series, n_values = ts_contaminated.shape
        start_index = int(math.ceil((n_values * self.FACTOR_SHIT)))
        population = (n_values - start_index)
        to_remove = int(math.ceil(population * missing_rate))

        if keep_other:
            for series in range(0, n_series):
                for col in range(population):
                    if series in series_selected:
                        if col <= to_remove:
                            ts_contaminated.iat[series, col+start_index] = np.nan
        else:
            for series in range(0, n_series):
                for col in range(population):
                        if col <= to_remove:
                            ts_contaminated.iat[series, col+start_index] = np.nan

        return ts_contaminated, ts_contaminated.to_numpy()