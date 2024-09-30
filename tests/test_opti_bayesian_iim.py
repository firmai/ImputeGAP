import unittest

from imputegap.recovery.contamination import Contamination
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.optimization import Optimization
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries



class TestOptiIMM(unittest.TestCase):

    def test_optimization_bayesian_iim(self):
        """
        the goal is to test if only the simple optimization with stmvl has the expected outcome
        """
        gap = TimeSeries(utils.get_file_path_dataset("chlorine"), limitation_values=100)

        algorithm = "iim"

        ts_contaminated = Contamination.scenario_mcar(ts=gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=2,
                                                      protection=0.1, use_seed=True, seed=42)

        optimal_params, yi = Optimization.Bayesian.bayesian_optimization(ground_truth=gap.ts,
                                                                         contamination=ts_contaminated,
                                                                         algorithm=algorithm, n_calls=2)

        print("\nOptimization done successfully... ")
        print("\n", optimal_params, "\n")

        params = utils.load_parameters(query="default", algorithm=algorithm)
        params_optimal = (optimal_params['learning_neighbors'], "iim 2")

        _, metrics_optimal = Imputation.Regression.iim_imputation(ground_truth=gap.ts, contamination=ts_contaminated, params=params_optimal)
        _, metrics_default = Imputation.Regression.iim_imputation(ground_truth=gap.ts, contamination=ts_contaminated, params=params)

        self.assertTrue(abs(metrics_optimal["RMSE"] - metrics_default["RMSE"]) < 0.1, f"Expected {metrics_optimal['RMSE']} > {metrics_default['RMSE']}")
        self.assertTrue(yi > 0, True)