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

        incomp_data = ts_1.Contamination.missing_completely_at_random(input_data=ts_1.data)


        algo = Imputation.DeepLearning.BitGraph(incomp_data).impute()  # user defined> or

        algo.incomp_data = incomp_data
        algo.score(input_data=ts_1.data, recov_data=algo.recov_data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 0.19658792092292338,
            "MAE": 0.16008693373135194,
            "MI": 0.18238122193233125,
            "CORRELATION": 0.3772340675711577
        }

        ts_1.print_results(algo.metrics, algo.algorithm)


        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2,
                       f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2,
                        f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3,
                        f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3,
                        f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")



    def test_imputation_bitgraph_udef(self):
        """
        the goal is to test if only the simple imputation with BitGraph has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.missing_completely_at_random(input_data=ts_1.data)

        algo = Imputation.DeepLearning.BitGraph(incomp_data).impute(user_def=True, params={"node_number":-1, "kernel_set":[1], "dropout":0.1, "subgraph_size":5, "node_dim":3, "seq_len":1, "lr":0.001, "epoch":2, "seed":42})  # user defined> or

        algo.incomp_data = incomp_data
        algo.score(input_data=ts_1.data, recov_data=algo.recov_data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 0.2107371692641671,
            "MAE": 0.1685821624473301,
            "MI": 0.17671310912326704,
            "CORRELATION": 0.4366175047451947
        }

        ts_1.print_results(algo.metrics, algo.algorithm)


        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.2,
                        f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.2,
                        f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3,
                        f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.3,
                        f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")

