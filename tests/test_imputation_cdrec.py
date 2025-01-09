import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestCDREC(unittest.TestCase):

    def test_imputation_cdrec(self):
        """
        the goal is to test if only the simple imputation with cdrec has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_timeseries(utils.search_path("test"))

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, series_rate=0.4, missing_rate=0.4, block_size=2, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.CDRec(incomp_data)
        algo.impute()
        algo.score(ts_1.data)

        _, metrics = algo.recov_data, algo.metrics

        expected_metrics = {
            "RMSE": 0.5993259196563864,
            "MAE": 0.5190054811092809,
            "MI": 0.7796564257088495,
            "CORRELATION": 0.6358270633906415
        }

        ts_1.print_results(metrics)

        assert np.isclose(metrics["RMSE"], expected_metrics["RMSE"]), f"RMSE mismatch: expected {expected_metrics['RMSE']}, got {metrics['RMSE']}"
        assert np.isclose(metrics["MAE"], expected_metrics["MAE"]), f"MAE mismatch: expected {expected_metrics['MAE']}, got {metrics['MAE']}"
        assert np.isclose(metrics["MI"], expected_metrics["MI"]), f"MI mismatch: expected {expected_metrics['MI']}, got {metrics['MI']}"
        assert np.isclose(metrics["CORRELATION"], expected_metrics["CORRELATION"]), f"Correlation mismatch: expected {expected_metrics['CORRELATION']}, got {metrics['CORRELATION']}"

    def test_imputation_cdrec_chlorine(self):
        """
        the goal is to test if only the simple imputation with cdrec has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_timeseries(utils.search_path("chlorine"), max_values=200)

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, series_rate=0.4, missing_rate=0.4, block_size=10, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.CDRec(incomp_data)
        algo.impute()
        algo.score(ts_1.data)

        _, metrics = algo.recov_data, algo.metrics

        expected_metrics = {
            "RMSE": 0.10329523970909142,
            "MAE": 0.06717112854576478,
            "MI": 0.7706445457837339,
            "CORRELATION": 0.913368365805848
        }

        ts_1.print_results(metrics)

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.1, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.1, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
