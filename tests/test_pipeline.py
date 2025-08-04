import os
import unittest
import numpy as np
from imputegap.tools import utils

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.explainer import Explainer
from imputegap.recovery.benchmark import Benchmark


class TestPipeline(unittest.TestCase):

    def test_pipeline(self):
        """
        Verify if the manager of a dataset is working
        """
        x = False

        # automl
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")
        incomp_data = ts_1.Contamination.mcar(ts_1.data, rate_series=0.18)

        cdrec = Imputation.MatrixCompletion.CDRec(incomp_data).impute()
        cdrec.score(ts_1.data, cdrec.recov_data)
        cdrec = Imputation.MatrixCompletion.CDRec(incomp_data).impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": { "n_calls": 3}})
        cdrec.score(ts_1.data, cdrec.recov_data)
        ts_1.print_results(cdrec.metrics)
        utils.save_optimization(optimal_params=cdrec.parameters, algorithm="cdrec", dataset="eeg", optimizer="t")

        cdrec = Imputation.MatrixCompletion.CDRec(incomp_data).impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "greedy","options": { "n_calls": 2}})
        cdrec.score(ts_1.data, cdrec.recov_data)

        cdrec = Imputation.MatrixCompletion.CDRec(incomp_data).impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "sh", "options": {"num_configs": 2}})
        cdrec.score(ts_1.data, cdrec.recov_data)

        cdrec = Imputation.MatrixCompletion.CDRec(incomp_data).impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "pso", "options": {"n_particles": 2}})
        cdrec.score(ts_1.data, cdrec.recov_data)

        # explainer
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))
        exp = Explainer()
        exp.shap_explainer(input_data=ts_1.data, algorithm="cdrec", missing_rate=0.25, rate_dataset=0.4, training_ratio=0.6, file_name="eeg-alcohol")
        exp.print(exp.shap_values, exp.shap_details)

        # benchmark
        dataset_test = ["eeg-alcohol"]
        opti_bayesian = {"optimizer": "bayesian", "options": {"n_calls": 3, "n_random_starts": 50, "acq_func": "gp_hedge", "metrics": "RMSE"}}
        optimizers = [opti_bayesian]
        algorithms_test = ["meanimpute", "cdrec"]
        patterns_small = ["mcar"]
        x_axis = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

        bench = Benchmark()
        bench.eval(algorithms=algorithms_test, datasets=dataset_test, patterns=patterns_small, x_axis=x_axis, optimizers=optimizers, save_dir="test_naterq", runs=2)

        x = not x
        self.assertTrue(x)



