import os
import unittest
import numpy as np

from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation
from imputegap.manager import utils
from imputegap.manager.manager import TimeSeries
from imputegap.optimization.bayesian_optimization import Optimization



class TestOptiIMM(unittest.TestCase):

    def test_optimization_bayesian_iim(self):
        """
        the goal is to test if only the simple optimization with stmvl has the expected outcome
        """
        gap = TimeSeries(utils.get_file_path_dataset("chlorine"))

        algorithm = "iim"

        ts_contaminated = Contamination.scenario_mcar(ts=gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=2,
                                                      protection=0.1, use_seed=True, seed=42)

        optimal_params, yi = Optimization.Bayesian.bayesian_optimization(ground_truth=gap.ts,
                                                                         contamination=ts_contaminated,
                                                                         algorithm=algorithm)

        print("\nOptimization done successfully... ")
        print("\n", optimal_params, "\n")

        params = Imputation.load_parameters(query="default", algorithm=algorithm)
        params_optimal = (optimal_params['neighbor'], "iim 2")

        _, metrics_optimal = Imputation.Regression.iim_imputation(ground_truth=gap.ts, contamination=ts_contaminated, params=params_optimal)
        _, metrics_default = Imputation.Regression.iim_imputation(ground_truth=gap.ts, contamination=ts_contaminated, params=params)

        Optimization.save_optimization(optimal_params=optimal_params, algorithm=algorithm+"_test")

        self.assertTrue(metrics_optimal["RMSE"] < metrics_default["RMSE"], f"Expected {metrics_optimal['RMSE']} > {metrics_default['RMSE']}")
        self.assertTrue(yi > 0, True)