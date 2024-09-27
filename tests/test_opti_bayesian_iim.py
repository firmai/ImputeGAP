import os
import unittest
import numpy as np

from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation
from imputegap.manager.manager import TimeSeries
from imputegap.optimization.bayesian_optimization import Optimization


def resolve_path(local_path, github_actions_path):
    """
    Find the accurate path for tests

    :param local_path: path of local code
    :param github_actions_path: path on GitHub action
    :return: correct file paths
    """
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(github_actions_path):
        return github_actions_path
    else:
        raise FileNotFoundError("File not found in both: ", local_path, " and ", github_actions_path)


def get_file_path(set_name="test"):
    """
    Find the accurate path for loading files of tests
    :return: correct file paths
    """
    return resolve_path(f'../imputegap/dataset/{set_name}.txt', f'./imputegap/dataset/{set_name}.txt')


class TestOptiIMM(unittest.TestCase):

    def test_optimization_bayesian_stmvl(self):
        """
        the goal is to test if only the simple optimization with stmvl has the expected outcome
        """
        gap = TimeSeries(get_file_path("chlorine"))

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

        _, metrics_optimal = Imputation.Regression.iim_imputation(ground_truth=gap.ts, contamination=ts_contaminated,
                                                 params=params_optimal)
        _, metrics_default = Imputation.Regression.iim_imputation(ground_truth=gap.ts, contamination=ts_contaminated, params=params)

        Optimization.save_optimization(optimal_params=optimal_params, algorithm=algorithm+"_test")

        self.assertTrue(metrics_optimal["RMSE"] > metrics_default["RMSE"], True)
        self.assertTrue(yi > 0, True)