import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries



class TestOptiMRNN(unittest.TestCase):

    def test_optimization_bayesian_mrnn(self):
        """
        the goal is to test if only the simple optimization with mrnn has the expected outcome
        """
        dataset, algorithm = "chlorine", "mrnn"

        ts_1 = TimeSeries()
        ts_1.load_series(data=utils.search_path(dataset), nbr_val=200)

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data)

        params = utils.load_parameters(query="default", algorithm=algorithm)

        algo_opti = Imputation.DeepLearning.MRNN(incomp_data)
        algo_opti.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 2}})

        algo_opti.score(input_data=ts_1.data)
        metrics_optimal = algo_opti.metrics

        algo_default = Imputation.DeepLearning.MRNN(incomp_data)
        algo_default.impute(params=params)
        algo_default.score(input_data=ts_1.data)
        metrics_default = algo_default.metrics

        print(f"{metrics_optimal = }")
        print(f"{metrics_default = }")


        self.assertTrue(abs(metrics_optimal["RMSE"] - metrics_default["RMSE"]) < 0.1, f"Expected {metrics_optimal['RMSE']} > {metrics_default['RMSE']}")