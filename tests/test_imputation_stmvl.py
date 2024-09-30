import unittest
import numpy as np

from imputegap.recovery.contamination import Contamination
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestSTMVL(unittest.TestCase):

    def test_imputation_mrnn_chlorine(self):
        """
        the goal is to test if only the simple imputation with ST-MVL has the expected outcome
        """
        impute_gap = TimeSeries(utils.get_file_path_dataset("chlorine"), limitation_values=200)

        ts_contaminated = Contamination.scenario_mcar(ts=impute_gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=10,
                                                      protection=0.1, use_seed=True, seed=42)

        imputation, metrics = Imputation.Pattern.stmvl_imputation(impute_gap.ts, ts_contaminated)

        expected_metrics = {
            "RMSE": 0.05795429338869703,
            "MAE": 0.038205100250362625,
            "MI": 0.9955907463544946,
            "CORRELATION": 0.9729604272282141
        }

        impute_gap.ts_contaminate = ts_contaminated
        impute_gap.ts_imputation = imputation
        impute_gap.metrics = metrics
        impute_gap.print_results()

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.1, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.1, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
