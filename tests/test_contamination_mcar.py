import os
import unittest
import numpy as np

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


def get_save_path():
    """
    Find the accurate path for saving files of tests
    :return: correct file paths
    """
    return resolve_path('../tests/assets', './tests/assets')


def get_file_path(set_name="test"):
    """
    Find the accurate path for loading files of tests
    :return: correct file paths
    """
    return resolve_path(f'../imputegap/dataset/{set_name}.txt', f'./imputegap/dataset/{set_name}.txt')


class TestContamination(unittest.TestCase):

    def test_mcar_selection(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        impute_gap = TimeSeries(get_file_path("test"))

        series_impacted = [0.4]
        missing_rates = [0.4]
        seeds_start, seeds_end = 42, 43
        series_check = ["8", "1", "5", "0"]
        protection = 0.1
        block_size = 2

        for seed_value in range(seeds_start, seeds_end):
            for series_sel in series_impacted:
                for missing_rate in missing_rates:

                    ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts,
                                                                 series_impacted=series_sel,
                                                                 missing_rate=missing_rate, block_size=block_size,
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

    def test_mcar_position(self):
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

                    ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts,
                                                                 series_impacted=series_sel,
                                                                 missing_rate=missing_rate,
                                                                 block_size=2, protection=0.1,
                                                                 use_seed=True, seed=seed_value)

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
        seeds_start, seeds_end = 42, 43
        protection = 0.1
        block_size = 10

        for dataset in datasets:
            impute_gap = TimeSeries(get_file_path(dataset))

            for seed_value in range(seeds_start, seeds_end):
                for series_sel in series_impacted:
                    for missing_rate in missing_rates:
                        ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts,
                                                                     missing_rate=missing_rate,
                                                                     series_impacted=series_sel,
                                                                     block_size=block_size, protection=protection,
                                                                     use_seed=True, seed=seed_value)

                        # 1) Check if the number of NaN values is correct
                        M, N = ts_contaminate.shape
                        P = int(N * protection)
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
        seeds_start, seeds_end = 42, 43
        protection = 0.1
        block_size = 10

        for dataset in datasets:
            impute_gap = TimeSeries(get_file_path(dataset))
            ten_percent_index = int(impute_gap.ts.shape[1] * 0.1)

            for seed_value in range(seeds_start, seeds_end):
                for series_sel in series_impacted:
                    for missing_rate in missing_rates:

                        ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts,
                                                                     series_impacted=series_sel,
                                                                     missing_rate=missing_rate,
                                                                     block_size=block_size, protection=protection,
                                                                     use_seed=True, seed=seed_value)

                        if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                            check_position = False
                        else:
                            check_position = True

                        self.assertTrue(check_position, True)

    def test_contaminate_plot(self):
        """
        Verify if the manager of a dataset is working
        """
        impute_gap = TimeSeries(get_file_path("chlorine"))
        impute_gap.ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts, series_impacted=0.4, missing_rate=0.1,
                                                                block_size=10, protection=0.1, use_seed=True, seed=42)

        impute_gap.print()
        filepath = impute_gap.plot("contamination", "test", get_save_path(), 5, (16, 8), False)

        self.assertTrue(os.path.exists(filepath))
