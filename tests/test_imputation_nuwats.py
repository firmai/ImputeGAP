import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestNuwaTS(unittest.TestCase):

    def test_imputation_nuwats_dft(self):
        """
        the goal is to test if only the simple imputation with NuwaTS has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.aligned(input_data=ts_1.data)

        algo = Imputation.LLMs.NuwaTS(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics = }")

        expected_metrics = {"RMSE": 0.6880447235513715, "MAE": 0.6462849827164852, "MI": 0.23292015433037982, "CORRELATION": 0.4529365138135433}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")

    def test_imputation_nuwats_user(self):
        """
        the goal is to test if only the simple imputation with NuwaTS has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.aligned(input_data=ts_1.data)

        algo = Imputation.LLMs.NuwaTS(incomp_data).impute(user_def=True, params={"seq_length":10, "patch_size":1, "batch_size":8, "pred_length":6, "label_length":10, "enc_in":10, "dec_in":10, "c_out": 10, "gpt_layers":4, "seed":42})
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics = }")

        expected_metrics = {"RMSE": 1.137214488264183, "MAE": 1.015034267832847, "MI": 0.5716802087840168, "CORRELATION": 0.6004926165059635}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")