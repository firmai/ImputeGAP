import unittest
import time

import numpy as np
from sklearn.impute import KNNImputer

from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestInterpolation(unittest.TestCase):

    def test_imputation_interpolation_chlorine(self):
        """
        the goal is to test if only the simple imputation with interpolation has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"), nbr_val=200)

        incomp_data = ts_1.Contamination.aligned(input_data=ts_1.data, rate_series=0.18)

        algo = Imputation.Statistics.Interpolation(incomp_data)
        algo.impute()
        algo.score(ts_1.data)

        _, metrics = algo.recov_data, algo.metrics

        expected_metrics = {"RMSE": 0.14768327523878277, "MAE": 0.10889031013542044, "MI": 0.782928657543708, "CORRELATION": 0.9039106910436514}

        ts_1.print_results(metrics)

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.1, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.1, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")