import unittest
import time

import numpy as np
from sklearn.impute import KNNImputer

from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestKNN(unittest.TestCase):

    def test_imputation_knn(self):
        """
        the goal is to test if only the simple imputation with IIM has the expected outcome
        """
        # 1. initiate the TimeSeries() object that will stay with you throughout the analysis

        k = 5
        weight = "uniform"

        ts_0 = TimeSeries()
        ts_0.load_series(utils.search_path("chlorine"))

        miss_ts = ts_0.Contamination.aligned(input_data=ts_0.data, rate_series=0.18, offset=0.1)

        imputer = Imputation.Statistics.KNNImpute(miss_ts)
        imputer.impute(user_def=True, params={"k":k, "weights":weight})
        imputer.score(ts_0.data, imputer.recov_data)

        ts_0.print_results(imputer.metrics, "knn", "imputegap lib")

        start_time = time.time()  # Record start time

        knn = KNNImputer(n_neighbors=k, weights=weight)
        recov = knn.fit_transform(miss_ts)

        imputer2 = Imputation.Statistics.KNNImpute(miss_ts)
        imputer2.recov_data = None
        imputer2.metrics = None

        imputer2.score(ts_0.data, np.array(recov))

        ts_0.print_results(imputer2.metrics, "knn", "sklearn lib")

        imputegap_metrics = imputer.metrics
        lib_metrics = imputer2.metrics

        end_time = time.time()

        print(f"\n\t\t> logs, imputation knn - Execution Time: {(end_time - start_time):.4f} seconds\n")


        self.assertTrue(abs(imputegap_metrics["RMSE"] - lib_metrics["RMSE"]) < 0.1, f"imputegap RMSE = {imputegap_metrics['RMSE']}, lib RMSE = {lib_metrics['RMSE']} ")
        self.assertTrue(abs(imputegap_metrics["MAE"] - lib_metrics["MAE"]) < 0.1, f"imputegap MAE = {imputegap_metrics['MAE']}, lib MAE = {lib_metrics['MAE']} ")
        self.assertTrue(abs(imputegap_metrics["MI"] - lib_metrics["MI"]) < 0.1, f"imputegap MI = {imputegap_metrics['MI']}, lib MI = {lib_metrics['MI']} ")
        self.assertTrue(abs(imputegap_metrics["CORRELATION"] - lib_metrics["CORRELATION"]) < 0.1, f"imputegap CORRELATION = {imputegap_metrics['CORRELATION']}, lib CORRELATION = {lib_metrics['CORRELATION']} ")