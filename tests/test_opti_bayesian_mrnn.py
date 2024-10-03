import unittest

from imputegap.recovery.contamination import Contamination
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.optimization import Optimization
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries



class TestOptiMRNN(unittest.TestCase):

    def test_optimization_bayesian_mrnn(self):
        """
        the goal is to test if only the simple optimization with mrnn has the expected outcome
        """
        gap = TimeSeries(utils.search_path("chlorine"), limitation_values=200)

        algorithm = "mrnn"

        ts_contaminated = Contamination.mcar(ts=gap.data, series_impacted=0.4, missing_rate=0.4, block_size=2, protection=0.1, use_seed=True, seed=42)

        optimal_params, yi = Optimization.Bayesian.bayesian_optimization(ground_truth=gap.data,
                                                                         contamination=ts_contaminated,
                                                                         algorithm=algorithm, n_calls=2)

        print("\nOptimization done successfully... ")
        print("\n", optimal_params, "\n")

        params = utils.load_parameters(query="default", algorithm=algorithm)
        _, _, _, seq = params
        params_optimal = (optimal_params['hidden_dim'], optimal_params['learning_rate'], optimal_params['iterations'], seq)

        _, metrics_optimal = Imputation.ML.mrnn_imputation(ground_truth=gap.data, contamination=ts_contaminated, params=params_optimal)
        _, metrics_default = Imputation.ML.mrnn_imputation(ground_truth=gap.data, contamination=ts_contaminated, params=params)

        self.assertTrue(abs(metrics_optimal["RMSE"] - metrics_default["RMSE"]) < 0.1, f"Expected {metrics_optimal['RMSE']} > {metrics_default['RMSE']}")
        self.assertTrue(yi > 0, True)