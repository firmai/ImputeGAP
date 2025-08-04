import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestSoftImpute(unittest.TestCase):

    def test_imputation_soft_impute_dft(self):
        """
        the goal is to test if only the simple imputation with SoftImpute has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.SoftImpute(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {"RMSE": 0.09627777349763414, "MAE": 0.07448970558861644, "MI": 0.7493079086248395, "CORRELATION": 0.8982101360155362}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.01, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.01, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.01, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.01, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")

    def test_imputation_soft_impute_udef(self):
        """
        the goal is to test if only the simple imputation with SoftImpute has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.SoftImpute(incomp_data).impute(params={"max_rank": 5})
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {"RMSE": 0.08170571441771922, "MAE": 0.06271339280403729, "MI": 0.911998312708481, "CORRELATION": 0.9345533571324787}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.01, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.01, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.01, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.01, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")