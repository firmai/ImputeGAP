import unittest

from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestOptiCDREC(unittest.TestCase):

    def test_optimization_bayesian_cdrec(self):
        """
        the goal is to test if only the simple optimization with cdrec has the expected outcome
        """
        algorithm = "cdrec"
        dataset = "chlorine"

        ts_1 = TimeSeries()
        ts_1.load_timeseries(utils.search_path(dataset))

        infected_matrix = ts_1.Contaminate.mcar(ts=ts_1.data, series_impacted=0.4, missing_rate=0.4, block_size=2, protection=0.1, use_seed=True, seed=42)

        params = utils.load_parameters(query="default", algorithm=algorithm)
        params_optimal_load = utils.load_parameters(query="optimal", algorithm=algorithm, dataset=dataset, optimizer="b")

        algo_opti = Imputation.MatrixCompletion.CDRec(infected_matrix)
        algo_opti.impute(user_defined=False, params={"ground_truth":ts_1.data, "optimizer":"bayesian", "options":{"n_calls": 2}})
        algo_opti.score(raw_matrix=ts_1.data)
        metrics_optimal = algo_opti.metrics

        algo_default = Imputation.MatrixCompletion.CDRec(infected_matrix)
        algo_default.impute(params=params)
        algo_default.score(raw_matrix=ts_1.data)
        metrics_default = algo_default.metrics

        algo_load = Imputation.MatrixCompletion.CDRec(infected_matrix)
        algo_load.impute(params=params_optimal_load)
        algo_load.score(raw_matrix=ts_1.data)
        metrics_optimal_load = algo_load.metrics

        self.assertTrue(metrics_optimal["RMSE"] < metrics_default["RMSE"], f"Expected {metrics_optimal['RMSE']} < {metrics_default['RMSE']} ")
        self.assertTrue(metrics_optimal_load["RMSE"] < metrics_default["RMSE"], f"Expected {metrics_optimal_load['RMSE']} < {metrics_default['RMSE']}")