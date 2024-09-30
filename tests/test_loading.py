import os
import unittest

import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestLoading(unittest.TestCase):

    def test_loading_set(self):
        """
        Verify if the manager of a dataset is working
        """
        impute_gap = TimeSeries(utils.get_file_path_dataset("test"))

        self.assertEqual(impute_gap.ts.shape, (10, 25))
        self.assertEqual(impute_gap.ts[0, 1], 2.5)
        self.assertEqual(impute_gap.ts[1, 0], 0.5)

    def test_loading_chlorine(self):
        """
        Verify if the manager of a dataset is working
        """
        impute_gap = TimeSeries(utils.get_file_path_dataset("chlorine"))

        self.assertEqual(impute_gap.ts.shape, (50, 1000))
        self.assertEqual(impute_gap.ts[0, 1], 0.0154797)
        self.assertEqual(impute_gap.ts[1, 0], 0.0236836)

    def test_loading_plot(self):
        """
        Verify if the manager of a dataset is working
        """
        impute_gap = TimeSeries(utils.get_file_path_dataset("test"))
        to_save = utils.get_save_path_asset()
        file_path = impute_gap.plot("gt", "test", to_save, 5, (16, 8), False)

        self.assertTrue(os.path.exists(file_path))

    def test_loading_normalization_min_max(self):
        impute_gap = TimeSeries(utils.get_file_path_dataset("test"), normalization="min_max")

        assert np.isclose(np.min(impute_gap.ts), 0), f"Min value after Min-Max normalization is not 0: {np.min(impute_gap.normalized_ts)}"
        assert np.isclose(np.max(impute_gap.ts), 1), f"Max value after Min-Max normalization is not 1: {np.max(impute_gap.normalized_ts)}"

    def test_loading_normalization_z_score(self):
        normalized = TimeSeries(utils.get_file_path_dataset("test"), normalization="z_score")

        mean = np.mean(normalized.ts)
        std_dev = np.std(normalized.ts)

        assert np.isclose(mean, 0, atol=1e-7), f"Mean after Z-score normalization is not 0: {mean}"
        assert np.isclose(std_dev, 1, atol=1e-7), f"Standard deviation after Z-score normalization is not 1: {std_dev}"

    def test_loading_normalization_min_max_lib(impute_gap):
        ground_truth = TimeSeries(utils.get_file_path_dataset("chlorine"))
        impute_gap = TimeSeries(utils.get_file_path_dataset("chlorine"), normalization="min_max")

        scaler = MinMaxScaler()
        lib_normalized = scaler.fit_transform(ground_truth.ts)

        assert np.allclose(impute_gap.ts, lib_normalized)

    def test_loading_normalization_z_score_lib(impute_gap):
        ground_truth = TimeSeries(utils.get_file_path_dataset("chlorine"))
        impute_gap = TimeSeries(utils.get_file_path_dataset("chlorine"), normalization="z_score")

        lib_normalized = zscore(ground_truth.ts, axis=None)

        assert np.allclose(impute_gap.ts, lib_normalized)

