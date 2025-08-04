import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestBRITS(unittest.TestCase):

    def test_imputation_brits_uni_dft(self):
        """
        the goal is to test if only the simple imputation with BRITS has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data)

        algo = Imputation.DeepLearning.BRITS(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics = }")

        expected_metrics = {"RMSE": 0.20352191729774288, "MAE": 0.16224711025280988, "MI": 0.20395832165069708, "CORRELATION": 0.34159910296350854}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.3, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.3, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.35, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.35, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")


    def test_imputation_brits_udef(self):
        """
        the goal is to test if only the simple imputation with BRITS has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.4, block_size=10, offset=0.1, seed=True)

        algo = Imputation.DeepLearning.BRITS(incomp_data).impute(params={"model": "brits", "epoch": 2, "batch_size": 10, "nbr_features": 1, "hidden_layer": 64, "num_workers": 0}, tr_ratio=0.8)
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics = }")

        expected_metrics = {"RMSE": 0.36766229001901735, "MAE": 0.3145462565890356, "MI": 0.03036856101641055, "CORRELATION": 0.01495670722764831}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.35, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.35, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.35, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.35, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")


    def test_imputation_brits_i_udef(self):
        """
        the goal is to test if only the simple imputation with BRITS has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data)

        algo = Imputation.DeepLearning.BRITS(incomp_data).impute(params={"model": "brits_i", "epoch": 2, "batch_size": 10, "nbr_features": 1, "hidden_layer": 64, "num_workers": 0})
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics = }")

        expected_metrics = {"RMSE": 0.34013719214536997, "MAE": 0.29081929660621414, "MI": 0.11204895391465408, "CORRELATION": 0.060251082945027054}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.3, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.3, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.35, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.35, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
