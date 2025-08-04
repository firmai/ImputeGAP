import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestHKMF_T(unittest.TestCase):

    def test_imputation_hkmf_t_dft(self):
        """
        the goal is to test if only the simple imputation with HKMF-T has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.aligned(ts_1.data, rate_dataset=0.2)

        algo = Imputation.DeepLearning.HKMF_T(incomp_data).impute()

        algo.score(ts_1.data)
        metrics = algo.metrics

        ts_1.print_results(algo.metrics, algo.algorithm)

        expected_metrics = { "RMSE": 0.2595099673955718, "MAE": 0.20747540959114055, "MI": 0.081000046038931, "CORRELATION": 0.1672221872103301 }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")


    def test_imputation_hkmf_t_udef(self):
        """
        the goal is to test if only the simple imputation with HKMF-T has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.aligned(ts_1.data, rate_dataset=0.2)

        algo = Imputation.DeepLearning.HKMF_T(incomp_data).impute(user_def=True, params={"tags":None, "data_names":None, "epoch":2})

        algo.score(ts_1.data, algo.recov_data)
        metrics = algo.metrics

        ts_1.print_results(algo.metrics, algo.algorithm)

        expected_metrics = { "RMSE": 0.2595099673955718, "MAE": 0.20747540959114055, "MI": 0.081000046038931, "CORRELATION": 0.1672221872103301 }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
