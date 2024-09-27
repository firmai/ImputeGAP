import os
import unittest
import numpy as np
import math

from imputegap.contamination.contamination import Contamination
from imputegap.manager.manager import TimeSeries


def resolve_path(local_path, github_actions_path):
    """
    Find the accurate path for tests

    :param local_path: path of local code
    :param github_actions_path: path on GitHub action
    :return: correct file paths
    """
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(github_actions_path):
        return github_actions_path
    else:
        raise FileNotFoundError("File not found in both: ", local_path, " and ", github_actions_path)


def get_file_path(set_name="test"):
    """
    Find the accurate path for loading files of tests
    :return: correct file paths
    """
    return resolve_path(f'../imputegap/dataset/{set_name}.txt', f'./imputegap/dataset/{set_name}.txt')


class TestContamination(unittest.TestCase):

    def test_mp_selection(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        impute_gap = TimeSeries(get_file_path("test"))

        series_impacted = [0.4]
        missing_rates = [0.4]
        seeds_start, seeds_end = 42, 43
        protection = 0.1

        length_of_gap_ts = len(impute_gap.ts[0])
        len_expected = math.ceil(missing_rates[0] * length_of_gap_ts)
        series_check = [str(i) for i in range(len_expected)]

        for seed_value in range(seeds_start, seeds_end):
            for series_sel in series_impacted:
                for missing_rate in missing_rates:


                    ts_contaminate = Contamination.scenario_missing_percentage(ts=impute_gap.ts,
                                                                 series_impacted=series_sel,
                                                                 missing_rate=missing_rate,
                                                                 protection=protection, use_seed=True,
                                                                 seed=seed_value)

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

    def test_mp_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        impute_gap = TimeSeries(get_file_path("test"))

        series_impacted = [0.4, 1]
        missing_rates = [0.1, 0.4, 0.6]
        ten_percent_index = int(impute_gap.ts.shape[1] * 0.1)
        seeds_start, seeds_end = 42, 43

        for seed_value in range(seeds_start, seeds_end):
            for series_sel in series_impacted:
                for missing_rate in missing_rates:

                    ts_contaminate = Contamination.scenario_missing_percentage(ts=impute_gap.ts,
                                                                 series_impacted=series_sel,
                                                                 missing_rate=missing_rate, protection=0.1,
                                                                 use_seed=True, seed=seed_value)

                    if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                        check_position = False
                    else:
                        check_position = True

                    self.assertTrue(check_position, True)