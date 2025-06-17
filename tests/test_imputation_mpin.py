import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestMPIN(unittest.TestCase):

    def test_imputation_mpin_dft(self):
        """
        the goal is to test if only the simple imputation with MPIN has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data)

        algo = Imputation.DeepLearning.MPIN(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics = }")

        expected_metrics = {"RMSE": 0.22420717576996374, "MAE": 0.1784756005505943, "MI": 0.34448439664289177, "CORRELATION": 0.6009656929810911 }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")



    def test_imputation_mpin_udef(self):
        """
        the goal is to test if only the simple imputation with MPIN has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data)

        algo = Imputation.DeepLearning.MPIN(incomp_data).impute(params={"incre_mode": "data+state", "window": 1, "k": 15, "learning_rate": 0.001, "weight_decay": 0.2, "epochs": 6, "num_of_iteration": 6, "threshold": 0.50, "base": "GCN"})
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics = }")

        expected_metrics = {"RMSE": 0.45674513439881553, "MAE": 0.3765008295901312, "MI": 0.09036730673278506, "CORRELATION": 0.06875929464207464 }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
