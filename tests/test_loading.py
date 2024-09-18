import os
import unittest

import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

from imputegap.manager._manager import TimeSeriesGAP


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

    def test_loading_normalization_min_max(self):
        impute_gap = TimeSeriesGAP(get_file_path("test"))
        impute_gap.normalization_min_max()

        assert np.isclose(np.min(impute_gap.normalized_ts), 0), f"Min value after Min-Max normalization is not 0: {np.min(impute_gap.normalized_ts)}"
        assert np.isclose(np.max(impute_gap.normalized_ts), 1), f"Max value after Min-Max normalization is not 1: {np.max(impute_gap.normalized_ts)}"

    def test_loading_normalization_z_score(self):
        impute_gap = TimeSeriesGAP(get_file_path("test"))
        impute_gap.normalization_z_score()

        mean = np.mean(impute_gap.normalized_ts)
        std_dev = np.std(impute_gap.normalized_ts)

        assert np.isclose(mean, 0, atol=1e-7), f"Mean after Z-score normalization is not 0: {mean}"
        assert np.isclose(std_dev, 1, atol=1e-7), f"Standard deviation after Z-score normalization is not 1: {std_dev}"

    def test_loading_normalization_min_max_lib(impute_gap):
        impute_gap = TimeSeriesGAP(get_file_path("chlorine"))
        impute_gap.normalization_min_max()

        scaler = MinMaxScaler()
        lib_normalized = scaler.fit_transform(impute_gap.ts)

        assert np.allclose(impute_gap.normalized_ts, lib_normalized)

    def test_loading_normalization_z_score_lib(impute_gap):
        impute_gap = TimeSeriesGAP(get_file_path("chlorine"))
        impute_gap.normalization_z_score()

        lib_normalized = zscore(impute_gap.ts, axis=None)

        assert np.allclose(impute_gap.normalized_ts, lib_normalized)

