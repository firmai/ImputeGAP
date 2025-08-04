import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestSTMVL(unittest.TestCase):

    def test_imputation_stmvl_chlorine(self):
        """
        the goal is to test if only the simple imputation with ST-MVL has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"), nbr_val=200)

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10,
                                              offset=0.1, seed=True)

        algo = Imputation.PatternSearch.STMVL(incomp_data)
        algo.impute()
        algo.score(ts_1.data)
        _, metrics = algo.recov_data, algo.metrics

        expected_metrics = {
            "RMSE": 0.1368300154471035,
            "MAE": 0.0826527606378556,
            "MI": 0.9622655017829951,
            "CORRELATION": 0.9324369637045151
        }

        ts_1.print_results(metrics)

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.1, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.1, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
