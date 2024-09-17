import os
import unittest

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


class TestLoading(unittest.TestCase):

    def test_loading_set(self):
        """
        Verify if the manager of a dataset is working
        """
        impute_gap = TimeSeriesGAP(get_file_path("test"))

        self.assertEqual(impute_gap.ts.shape, (10, 25))
        self.assertEqual(not impute_gap.filename, False)
        self.assertEqual(impute_gap.ts[0, 1], 2.5)
        self.assertEqual(impute_gap.ts[1, 0], 0.5)

    def test_loading_chlorine(self):
        """
        Verify if the manager of a dataset is working
        """
        impute_gap = TimeSeriesGAP(get_file_path("chlorine"))

        self.assertEqual(impute_gap.ts.shape, (50, 1000))
        self.assertEqual(not impute_gap.filename, False)
        self.assertEqual(impute_gap.ts[0, 1], 0.0154797)
        self.assertEqual(impute_gap.ts[1, 0], 0.0236836)

    def test_loading_plot(self):
        """
        Verify if the manager of a dataset is working
        """
        impute_gap = TimeSeriesGAP(get_file_path("test"))
        impute_gap.plot("ground_truth", "test", get_save_path(), 5, (16, 8), False)

        self.assertTrue(os.path.exists(get_save_path()))
