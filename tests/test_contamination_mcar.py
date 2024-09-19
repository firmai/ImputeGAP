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


        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        seeds_start, seeds_end = 42, 43

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts, missing_rate=missing_rate, block_size=2, series_selected=series_selected, starting_position=0.1, use_seed=True, seed=seed_value)
                    series_check = Contamination.format_selection(impute_gap.ts, series_selected)

                    impute_gap.print()

                    check_nan_series = False

                    for series, data in enumerate(impute_gap.ts_contaminate):
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

        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        ten_percent_index = int(impute_gap.ts.shape[1] * 0.1)
        seeds_start, seeds_end = 42, 43

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts, missing_rate=missing_rate, block_size=2, series_selected=series_selected, starting_position=0.1, use_seed=True, seed=seed_value)

                    if np.isnan(impute_gap.ts_contaminate[:, :ten_percent_index]).any():
                        check_position = False
                    else:
                        check_position = True

                    self.assertTrue(check_position, True)

    def test_mcar_selection_datasets(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        datasets = ["bafu", "chlorine", "climate", "drift", "meteo"]
        series_selection = [["1", "3", "4"], ["+2"], ["*"]]
        missing_rates = [0.2, 0.4, 0.8]
        seeds_start, seeds_end = 42, 43

        for dataset in datasets:
            impute_gap = TimeSeries(get_file_path(dataset))

            for seed_value in range(seeds_start, seeds_end):
                for series_selected in series_selection:
                    for missing_rate in missing_rates:

                        impute_gap.ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts, missing_rate=missing_rate, block_size=10, series_selected=series_selected, starting_position=0.1, use_seed=True, seed=seed_value)
                        series_check = Contamination.format_selection(impute_gap.ts, series_selected)

                        check_nan_series = False

                        for series, data in enumerate(impute_gap.ts_contaminate):
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

    def test_mcar_position_datasets(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        datasets = ["bafu", "chlorine", "climate", "drift", "meteo"]
        series_selection = [["1", "3", "4"], ["+2"], ["*"]]
        missing_rates = [0.2, 0.4, 0.8]
        seeds_start, seeds_end = 42, 43

        for dataset in datasets:
            impute_gap = TimeSeries(get_file_path(dataset))
            ten_percent_index = int(impute_gap.ts.shape[1] * 0.1)

            for seed_value in range(seeds_start, seeds_end):
                for series_selected in series_selection:
                    for missing_rate in missing_rates:

                        impute_gap.ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts, missing_rate=missing_rate, block_size=10, series_selected=series_selected, starting_position=0.1, use_seed=True, seed=seed_value)

                        if np.isnan(impute_gap.ts_contaminate[:, :ten_percent_index]).any():
                            check_position = False
                        else:
                            check_position = True

                        self.assertTrue(check_position, True)

    def test_contaminate_plot(self):
        """
        Verify if the manager of a dataset is working
        """
        # if not hasattr(matplotlib.get_backend(), 'required_interactive_framework'):
        #    matplotlib.use('Agg')
        impute_gap = TimeSeries(get_file_path("chlorine"))
        impute_gap.ts_contaminate = Contamination.scenario_mcar(ts=impute_gap.ts, missing_rate=0.1, block_size=10, series_selected=["1"], starting_position=0.1, use_seed=True)

        impute_gap.plot("contamination", "test", get_save_path(), 5, (16, 8), False)

        self.assertTrue(os.path.exists(get_save_path()))
