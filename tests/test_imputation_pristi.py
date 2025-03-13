import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestPRISTI(unittest.TestCase):

    def test_imputation_pristi_dft(self):
        """
        the goal is to test if only the simple imputation with PRISTI has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.missing_completely_at_random(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10,
                                                                      offset=0.1, seed=True)

        algo = Imputation.DeepLearning.PRISTI(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 56.927179029290606,
            "MAE": 45.72093928559217,
            "MI": 0.024087511524174616,
            "CORRELATION": -0.01583714423419342
        }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 1.1,
                       f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 1.1,
                        f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 1.3,
                        f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 1.3,
                        f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")


    def test_imputation_pristi_udef(self):
        """
        the goal is to test if only the simple imputation with PRISTI has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.missing_completely_at_random(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10,
                                                                      offset=0.1, seed=True)

        algo = Imputation.DeepLearning.PRISTI(incomp_data).impute(params={"target_strategy":"hybrid", "unconditional":True, "seed":42, "device":"cpu"})
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 56.92715121574767,
            "MAE": 45.72093928559217,
            "MI": 0.024087511524174616,
            "CORRELATION": -0.01583714423419342
        }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 1.1,
                       f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 1.1,
                        f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 1.3,
                        f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 1.3,
                        f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
