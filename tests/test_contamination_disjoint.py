import unittest
import numpy as np
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestContaminationDisjoint(unittest.TestCase):

    def test_disjoint_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))

        series_impacted = [0.4, 0.8]
        ten_percent_index = int(ts_1.data.shape[1] * 0.1)

        for series_sel in series_impacted:

            ts_contaminate = ts_1.Contamination.disjoint(input_data=ts_1.data, series_rate=series_sel, limit=1, offset=0.1)

            if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                check_position = False
            else:
                check_position = True

            self.assertTrue(check_position, True)