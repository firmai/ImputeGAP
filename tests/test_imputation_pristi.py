import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestPRISTI(unittest.TestCase):

    def test_imputation_pristi_dft(self):
        """
        the goal is to test if only the simple imputation with PRISTI has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"), nbr_series=200)
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data)

        algo = Imputation.DeepLearning.PRISTI(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        print(f"{metrics = }")

        expected_metrics = {"RMSE": 0.6016833787111799, "MAE": 0.5139590505163977, "MI": 0.07591969788159918, "CORRELATION": -0.020170456468113926}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.4, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.4, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")