import unittest
import numpy as np
import math
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestContamination(unittest.TestCase):

    def test_mp_selection(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        ts_1 = TimeSeries()
        ts_1.load_timeseries(utils.search_path("test"))

        series_impacted = [0.4, 1]
        missing_rates = [0.4, 1]
        protection = 0.1
        M, N = ts_1.data.shape

        for series_per in series_impacted:
            for missing_rate in missing_rates:
                ts_contaminate = ts_1.Contaminate.missing_percentage(ts=ts_1.data,
                                                                  series_impacted=series_per,
                                                                  missing_rate=missing_rate,
                                                                  protection=protection)

                n_nan = np.isnan(ts_contaminate).sum()
                expected_nan_series = math.ceil(series_per * M)
                expected_nan_values = int((N - int(N * protection)) * missing_rate)
                expected = expected_nan_series * expected_nan_values

                self.assertEqual(n_nan, expected, f"Expected {expected} contaminated series but found {n_nan}")

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

                ts_contaminate = ts_1.Contaminate.missing_percentage(ts=ts_1.data,
                                                                  series_impacted=series_sel,
                                                                  missing_rate=missing_rate, protection=0.1)

                if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                    check_position = False
                else:
                    check_position = True

                self.assertTrue(check_position, True)