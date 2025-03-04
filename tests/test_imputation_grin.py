import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestGRIN(unittest.TestCase):

    def test_imputation_grin_dft(self):
        """
        the goal is to test if only the simple imputation with GRIN has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(ts_1.data, rate_series=0.18)

        algo = Imputation.DeepLearning.GRIN(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        ts_1.print_results(algo.metrics, algo.algorithm)

        expected_metrics = { "RMSE": 0.22863432711255152, "MAE": 0.178475889298036, "MI": 0.15910943933971244, "CORRELATION": 0.2949412054264574 }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1,
                       f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1,
                        f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3,
                        f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3,
                        f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")


    def test_imputation_grin_udef(self):
        """
        the goal is to test if only the simple imputation with GRIN has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(ts_1.data, rate_series=0.18)

        algo = Imputation.DeepLearning.GRIN(incomp_data).impute(user_def=True, params={"d_hidden":32, "lr":0.001, "batch_size":32, "window":1, "alpha":10.0, "patience":4, "epochs":20, "workers":2})

        algo.score(ts_1.data)
        metrics = algo.metrics

        ts_1.print_results(algo.metrics, algo.algorithm)

        expected_metrics = { "RMSE": 0.22863432711255152, "MAE": 0.178475889298036, "MI": 0.15910943933971244, "CORRELATION": 0.2949412054264574 }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1,
                       f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1,
                        f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3,
                        f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3,
                        f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
