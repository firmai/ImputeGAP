import os
import unittest
import numpy as np
from imputegap.manager._manager import TimeSeriesGAP


def resolve_path(local_path, github_actions_path):
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(github_actions_path):
        return github_actions_path
    else:
        raise FileNotFoundError(f"File not found in both: {local_path} and {github_actions_path}.")


def get_save_path():
    return resolve_path('../imputegap/assets', './imputegap/assets')


def get_file_path(set_name="test"):
    return resolve_path(f'../imputegap/dataset/{set_name}.txt', f'./imputegap/dataset/{set_name}.txt')


class TestContamination(unittest.TestCase):

    def test_mcar_selection(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        impute_gap = TimeSeriesGAP(get_file_path("test"))

        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        seeds_start, seeds_end = 42, 43

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.contamination_mcar(missing_rate=missing_rate, block_size=2, starting_position=0.1,
                                                  series_selected=series_selected, use_seed=True, seed=seed_value)

                    series_check = impute_gap.format_selection(series_selected)

                    impute_gap.print()

                    check_nan_series = False

                    for series, data in enumerate(impute_gap.contaminated_ts):
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
        impute_gap = TimeSeriesGAP(get_file_path("test"))

        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        ten_percent_index = int(impute_gap.ts.shape[1] * 0.1)
        seeds_start, seeds_end = 42, 43

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.contamination_mcar(missing_rate=missing_rate, block_size=2, starting_position=0.1,
                                                  series_selected=series_selected, use_seed=True, seed=seed_value)

                    impute_gap.print()

                    if np.isnan(impute_gap.contaminated_ts[:, :ten_percent_index]).any():
                        check_position = False
                    else:
                        check_position = True

                    self.assertTrue(check_position, True)

    def test_mcar_selection_chlorine(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        impute_gap = TimeSeriesGAP(get_file_path("chlorine"))

        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        seeds_start, seeds_end = 42, 43

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.contamination_mcar(missing_rate=missing_rate, block_size=10, starting_position=0.1,
                                                  series_selected=series_selected, use_seed=True, seed=seed_value)

                    series_check = impute_gap.format_selection(series_selected)

                    impute_gap.print()

                    check_nan_series = False

                    for series, data in enumerate(impute_gap.contaminated_ts):
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

    def test_mcar_position_chlorine(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        impute_gap = TimeSeriesGAP(get_file_path("chlorine"))

        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        seeds_start, seeds_end = 42, 43

        ten_percent_index = int(impute_gap.ts.shape[1] * 0.1)

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.contamination_mcar(missing_rate=missing_rate, block_size=10, starting_position=0.1,
                                                  series_selected=series_selected, use_seed=True, seed=seed_value)

                    impute_gap.print()

                    if np.isnan(impute_gap.contaminated_ts[:, :ten_percent_index]).any():
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

        impute_gap = TimeSeriesGAP(get_file_path("chlorine"))

        impute_gap.plot("contaminate", "test", get_save_path(), 5, (16, 8), False)

        self.assertTrue(os.path.exists(get_save_path()))
