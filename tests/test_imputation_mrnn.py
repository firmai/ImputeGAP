import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestMRNN(unittest.TestCase):

    def test_imputation_mrnn_chlorine(self):
        """
        the goal is to test if only the simple imputation with MRNN has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"), nbr_val=200)

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10, offset=0.1, seed=True)

        algo = Imputation.DeepLearning.MRNN(incomp_data)
        algo.impute(tr_ratio=0.8)
        algo.score(ts_1.data)

        _, metrics = algo.recov_data, algo.metrics

        expected_metrics = {
            "RMSE": 0.2605172813727444,
            "MAE": 0.16088221520455398,
            "MI": 0.29905669450152045,
            "CORRELATION": 0.404769322095035
        }

        print(f"{metrics = }")

        ts_1.print_results(metrics)

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.3, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.35, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
