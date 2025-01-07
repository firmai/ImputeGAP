import os
import unittest
import numpy as np
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestContamination(unittest.TestCase):

    def test_mcar_selection(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        ts_1 = TimeSeries()
        ts_1.load_timeseries(utils.search_path("test"))

        series_impacted = [0.4]
        missing_rates = [40]
        series_check = ["8", "1", "5", "0"]
        offset = 0.1
        block_size = 2


        for series_sel in series_impacted:
            for missing_rate in missing_rates:

                ts_contaminate = ts_1.Contamination.mcar(input_data=ts_1.data,
                                                         series_rate=series_sel,
                                                         missing_rate=missing_rate, block_size=block_size,
                                                         offset=offset, seed=True)

                check_nan_series = False

                for series, data in enumerate(ts_contaminate):
                    if str(series) in series_check:
                        if np.isnan(data).any():
                            check_nan_series = True
                    else:
                        if np.isnan(data).any():
                            check_nan_series = False
                            break
                        else:
                            check_nan_series = True

                self.assertTrue(check_nan_series, True)

    def test_mcar_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        ts_1 = TimeSeries()
        ts_1.load_timeseries(utils.search_path("test"))

        series_impacted = [0.4, 1]
        missing_rates = [0.1, 0.4, 0.6]
        ten_percent_index = int(ts_1.data.shape[1] * 0.1)

        for series_sel in series_impacted:
            for missing_rate in missing_rates:

                ts_contaminate = ts_1.Contamination.mcar(input_data=ts_1.data,
                                                         series_rate=series_sel,
                                                         missing_rate=missing_rate,
                                                         block_size=2, offset=0.1,
                                                         seed=True)

                if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                    check_position = False
                else:
                    check_position = True

                self.assertTrue(check_position, True)

    def test_mcar_selection_datasets(self):
        """
        test if only the selected values are contaminated in the right % of series with the right amount of values
        """
        datasets = ["bafu", "chlorine", "climate", "drift", "meteo"]
        series_impacted = [0.4, 1]
        missing_rates = [0.2, 0.6]
        offset = 0.1
        block_size = 10

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.load_timeseries(utils.search_path(dataset))


            for series_sel in series_impacted:
                for missing_rate in missing_rates:
                    ts_contaminate = ts_1.Contamination.mcar(input_data=ts_1.data,
                                                             missing_rate=missing_rate,
                                                             series_rate=series_sel,
                                                             block_size=block_size, offset=offset,
                                                             seed=True)

                    # 1) Check if the number of NaN values is correct
                    M, N = ts_contaminate.shape
                    P = int(N * offset)
                    W = int((N - P) * missing_rate)
                    expected_contaminated_series = int(np.ceil(M * series_sel))
                    B = int(W / block_size)
                    total_expected = (B * block_size) * expected_contaminated_series
                    total_nan = np.isnan(ts_contaminate).sum()

                    self.assertEqual(total_nan, total_expected)

                    # 2) Check if the correct percentage of series are contaminated
                    contaminated_series = np.isnan(ts_contaminate).any(axis=1).sum()
                    self.assertEqual(contaminated_series, expected_contaminated_series, f"Expected {expected_contaminated_series} contaminated series but found {contaminated_series}")

    def test_mcar_position_datasets(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        datasets = ["bafu", "chlorine", "climate", "drift", "meteo"]
        series_impacted = [0.4, 1]
        missing_rates = [0.2, 0.6]
        offset = 0.1
        block_size = 10

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.load_timeseries(utils.search_path(dataset))
            ten_percent_index = int(ts_1.data.shape[1] * 0.1)

            for series_sel in series_impacted:
                for missing_rate in missing_rates:

                    ts_contaminate = ts_1.Contamination.mcar(input_data=ts_1.data,
                                                             series_rate=series_sel,
                                                             missing_rate=missing_rate,
                                                             block_size=block_size, offset=offset,
                                                             seed=True)

                    if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                        check_position = False
                    else:
                        check_position = True

                    self.assertTrue(check_position, True)

    def test_contaminate_plot(self):
        """
        Verify if the manager of a dataset is working
        """
        ts_1 = TimeSeries()
        ts_1.load_timeseries(utils.search_path("chlorine"))

        ts_2 = TimeSeries()
        ts_2.import_matrix(ts_1.Contamination.mcar(input_data=ts_1.data, series_rate=0.4, missing_rate=0.1,
                                                   block_size=10, offset=0.1, seed=True))

        ts_1.print()
        filepath = ts_1.plot(input_data=ts_1.data, incomp_data=ts_2.data, max_series=10, max_values=100, save_path="./assets/", display=False)
        self.assertTrue(os.path.exists(filepath))

    def test_mcar_size_of_block(self):
        """
        test if the size of the block is at least the number defined my the user
        """
        datasets = ["drift", "chlorine", "eeg-reading", "eeg-alcohol", "fmri-objectviewing", "fmri-stoptask"]
        series_impacted = [0.4, 1]
        missing_rates = [0.2, 0.6]
        offset = 0.1
        block_size = 10

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.load_timeseries(utils.search_path(dataset))

            for series_sel in series_impacted:
                for missing_rate in missing_rates:
                    ts_contaminate = ts_1.Contamination.mcar(input_data=ts_1.data,
                                                             missing_rate=missing_rate,
                                                             series_rate=series_sel,
                                                             block_size=block_size, offset=offset,
                                                             seed=True)

                    for i, series in enumerate(ts_contaminate):
                        nan_blocks = []
                        block_indices = []
                        current_block_size = 0
                        series_size = len(series)
                        lower_bound = int(offset * series_size) + block_size
                        upper_bound = series_size - lower_bound - block_size
                        protected_indices = set(range(0, lower_bound)) | set(range(upper_bound, series_size))

                        # Find NaN blocks and their indices
                        for index, value in enumerate(series):
                            if np.isnan(value):
                                current_block_size += 1
                                block_indices.append(index)
                            else:
                                if current_block_size > 0:
                                    if not any(i in protected_indices for i in block_indices):
                                        nan_blocks.append(current_block_size)
                                    current_block_size = 0
                                    block_indices = []

                        for block in nan_blocks:
                            assert block >= block_size, (
                                f"Dataset: {dataset}, Series: {i}, Seed: {seed_value}, "
                                f"Block size {block} found, expected at least {block_size}."
                            )