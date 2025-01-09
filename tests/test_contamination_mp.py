import unittest
import numpy as np
import math
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestContamination(unittest.TestCase):

    def test_mp_selection(self):
        """
        the goal is to test if the number of NaN values expected are provided in the contamination output
        """

        datasets = ["drift", "chlorine", "eeg-alcohol", "fmri-objectviewing", "fmri-stoptask"]
        series_impacted = [0.1, 0.5, 1]  # percentage of series impacted
        missing_rates = [0.1, 0.5, 1]  # percentage of missing values with NaN
        P = 0.1  # offset zone

        for dataset in datasets:
            ts = TimeSeries()
            ts.load_timeseries(utils.search_path(dataset))
            M, N = ts.data.shape  # series, values

            for S in series_impacted:
                for R in missing_rates:
                    incomp_data = ts.Contamination.missing_percentage(input_data=ts.data, series_rate=S, missing_rate=R, offset=P)

                    n_nan = np.isnan(incomp_data).sum()
                    expected_nan_series = math.ceil(S * M)
                    expected_nan_values = int((N - int(N * P)) * R)
                    expected_nan = expected_nan_series * expected_nan_values

                    self.assertEqual(n_nan, expected_nan, f"Expected {expected_nan} contaminated series but found {n_nan}")

    def test_mp_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        ts_1 = TimeSeries()
        ts_1.load_timeseries(utils.search_path("test"))

        series_impacted = [0.4, 0.8]
        missing_rates = [0.1, 0.4, 0.6]
        ten_percent_index = int(ts_1.data.shape[1] * 0.1)

        for series_sel in series_impacted:
            for missing_rate in missing_rates:

                ts_contaminate = ts_1.Contamination.missing_percentage(input_data=ts_1.data,
                                                                       series_rate=series_sel,
                                                                       missing_rate=missing_rate, offset=0.1)

                if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                    check_position = False
                else:
                    check_position = True

                self.assertTrue(check_position, True)