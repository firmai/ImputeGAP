import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestXGBOOST(unittest.TestCase):

    def test_imputation_xgboost_dft(self):
        """
        the goal is to test if only the simple imputation with XGBOOST has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10,
                                              offset=0.1, seed=True)

        algo = Imputation.MachineLearning.XGBOOST(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 0.17754488848918099,
            "MAE": 0.13078221341470944,
            "MI": 0.34075635474360655,
            "CORRELATION": 0.6071452702522209
        }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1,
                       f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1,
                        f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.2,
                        f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.2,
                        f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")