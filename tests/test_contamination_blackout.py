import unittest
import numpy as np
import math

from imputegap.recovery.contamination import Contamination
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestContamination(unittest.TestCase):

    def test_mp_selection(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        gap = TimeSeries(utils.get_file_path_dataset("test"))

        missing_rates = [0.4, 1]
        protection = 0.1
        M, N = gap.ts.shape

        for missing_rate in missing_rates:
            ts_contaminate = Contamination.scenario_blackout(ts=gap.ts, missing_rate=missing_rate, protection=protection)

            n_nan = np.isnan(ts_contaminate).sum()
            expected_nan_series = M
            expected_nan_values = int((N - int(N * protection)) * missing_rate)
            expected = expected_nan_series * expected_nan_values

            self.assertEqual(n_nan, expected, f"Expected {expected} contaminated series but found {n_nan}")

    def test_mp_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        impute_gap = TimeSeries(utils.get_file_path_dataset("test"))

        missing_rates = [0.1, 0.4, 0.6]
        ten_percent_index = int(impute_gap.ts.shape[1] * 0.1)

        for missing_rate in missing_rates:

            ts_contaminate = Contamination.scenario_blackout(ts=impute_gap.ts, missing_rate=missing_rate, protection=0.1)

            if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                check_position = False
            else:
                check_position = True

            self.assertTrue(check_position, True)