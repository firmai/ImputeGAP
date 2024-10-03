import unittest

from imputegap.recovery.contamination import Contamination
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.optimization import Optimization
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestOptiCDREC(unittest.TestCase):

    def test_optimization_bayesian_cdrec(self):
        """
        the goal is to test if only the simple optimization with cdrec has the expected outcome
        """
        algorithm = "cdrec"
        dataset = "chlorine"

        gap = TimeSeries(utils.search_path(dataset))

        ts_contaminated = Contamination.mcar(ts=gap.data, series_impacted=0.4, missing_rate=0.4, block_size=2, protection=0.1, use_seed=True, seed=42)

        optimal_params, yi = Optimization.Bayesian.bayesian_optimization(ground_truth=gap.data, contamination=ts_contaminated, algorithm=algorithm, n_calls=3)

        print("\nOptimization done successfully... ")
        print("\n", optimal_params, "\n")

        params = utils.load_parameters(query="default", algorithm=algorithm)
        params_optimal = (optimal_params['rank'], optimal_params['epsilon'], optimal_params['iteration'])
        params_optimal_load = utils.load_parameters(query="optimal", algorithm=algorithm, dataset=dataset, optimizer="b")


        _, metrics_optimal = Imputation.MD.cdrec(ground_truth=gap.data, contamination=ts_contaminated, params=params_optimal)
        _, metrics_default = Imputation.MD.cdrec(ground_truth=gap.data, contamination=ts_contaminated, params=params)
        _, metrics_optimal_load = Imputation.MD.cdrec(ground_truth=gap.data, contamination=ts_contaminated, params=params_optimal_load)

        self.assertTrue(metrics_optimal["RMSE"] < metrics_default["RMSE"], f"Expected {metrics_optimal['RMSE']} < {metrics_default['RMSE']} ")
        self.assertTrue(metrics_optimal_load["RMSE"] < metrics_default["RMSE"], f"Expected {metrics_optimal_load['RMSE']} < {metrics_default['RMSE']}")
        self.assertTrue(yi > 0, True)