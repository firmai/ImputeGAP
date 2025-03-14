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
        ts_1.load_series(utils.search_path("test"))

        incomp_data = ts_1.Contamination.missing_completely_at_random(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=2, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.CDRec(incomp_data)
        algo.impute()
        algo.score(ts_1.data)

        _, metrics = algo.recov_data, algo.metrics

        expected_metrics = {
            "RMSE": 0.4345469663511766,
            "MAE": 0.364996518101561,
            "MI": 1.1044925396596248,
            "CORRELATION": 0.794760428131731
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
        ts_1.load_series(utils.search_path("chlorine"), nbr_val=200)

        incomp_data = ts_1.Contamination.missing_completely_at_random(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.CDRec(incomp_data)
        algo.impute()
        algo.score(ts_1.data)

        _, metrics = algo.recov_data, algo.metrics

        expected_metrics = {
            "RMSE": 0.07467415556012959,
            "MAE": 0.04927307586281738,
            "MI": 0.9032246175289653,
            "CORRELATION": 0.9583571591921054
        }

        ts_1.print_results(metrics)

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.1, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.1, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
