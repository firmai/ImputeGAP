import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestBitGraph(unittest.TestCase):

    def test_imputation_bitgraph_dft(self):
        """
        the goal is to test if only the simple imputation with BitGraph has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data)


        algo = Imputation.DeepLearning.BitGraph(incomp_data).impute()  # user defined> or

        algo.incomp_data = incomp_data
        algo.score(input_data=ts_1.data, recov_data=algo.recov_data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 0.20218134150858114,
            "MAE": 0.16291651426264953,
            "MI": 0.1489051542044696,
            "CORRELATION": 0.3801492668587883
        }

        ts_1.print_results(algo.metrics, algo.algorithm)


        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")

