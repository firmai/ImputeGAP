import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestDeepMVI(unittest.TestCase):

    def test_imputation_deep_mvi_dft(self):
        """
        the goal is to test if only the simple imputation with DeepMVI has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data)

        algo = Imputation.DeepLearning.DeepMVI(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics}")

        expected_metrics = {
            "RMSE": 0.14590414996535767,
            "MAE": 0.10577546184718635,
            "MI": 0.4777684710034327,
            "CORRELATION": 0.7433559069460258
        }

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1,
                       f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1,
                        f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3,
                        f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3,
                        f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")