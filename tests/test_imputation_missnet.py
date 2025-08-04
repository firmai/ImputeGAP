import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestMissNet(unittest.TestCase):

    def test_imputation_missnet_dft(self):
        """
        the goal is to test if only the simple imputation with MissNet has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"), nbr_series=40, nbr_val=100)
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(ts_1.data)

        algo = Imputation.DeepLearning.MissNet(incomp_data).impute(
            params={'alpha': 0.5, 'beta': 0.1, 'L': 10, 'n_cl': 1, 'max_iteration': 3, 'tol': 2, 'random_init': False})
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics = }")

        expected_metrics = {"RMSE": 0.11351861978049634, "MAE": 0.09000628925874102, "MI": 0.7766779449403466, "CORRELATION": 0.881484416469724}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")