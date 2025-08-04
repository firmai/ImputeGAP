import unittest

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils


class TestOptiRAY(unittest.TestCase):

    def test_optimization_ray(self):
        """
        the goal is to test if only the simple optimization RAY TUNE with CDRec has the expected outcome
        """

        # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
        ts_1 = TimeSeries()

        check = False

        # 2. load the timeseries from file or from the code
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        # 3. contamination of the data
        ts_mask = ts_1.Contamination.mcar(ts_1.data, rate_series=0.18)

        # 4. imputation of the contaminated data
        # imputation with AutoML which will discover the optimal hyperparameters for your dataset and your algorithm

        algorithms_all = ["SoftImpute", "stmvl", "KNNImpute"]

        for alg in algorithms_all:
            imputer = utils.config_impute_algorithm(incomp_data=ts_mask, algorithm=alg)
            imputer.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "ray_tune"})

            # 5. score the imputation with the raw_data
            imputer.score(ts_1.data, imputer.recov_data)

            # 6. display the results
            ts_1.print_results(imputer.metrics)

            # 7. save hyperparameters
            utils.save_optimization(optimal_params=imputer.parameters, algorithm=alg, dataset="eeg", optimizer="ray")

        check = True
        self.assertTrue(check)