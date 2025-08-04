import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestGAIN(unittest.TestCase):

    def test_imputation_gain_dft(self):
        """
        the goal is to test if only the simple imputation with GAIN has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(ts_1.data, rate_series=0.18)

        algo = Imputation.DeepLearning.GAIN(incomp_data).impute()

        algo.score(ts_1.data)
        metrics = algo.metrics

        ts_1.print_results(algo.metrics, algo.algorithm)

        expected_metrics = { "RMSE": 0.09343311918947016, "MAE": 0.0707023385813999, "MI": 0.7875021240298573, "CORRELATION": 0.8890104658851162 }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.4, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.4, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")