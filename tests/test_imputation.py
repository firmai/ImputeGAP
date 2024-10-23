import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestImputation(unittest.TestCase):

    def test_imputation_min(self):
        """
        the goal is to test if only the simple imputation with min has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.import_matrix(np.array([[1, 2, 1], [4, 2, 6]]))

        contamination = np.array([[1, 2, np.nan], [4, np.nan, 6]])

        algo = Imputation.Statistics.MinImpute(contamination)
        algo.impute()
        algo.score(ts_1.data)

        result = np.array([[1, 2, 1], [4, 1, 6]])

        imputation, metrics = algo.imputed_matrix, algo.metrics

        assert np.all(np.isclose(imputation, result)), f"imputation: expected {result}, got {imputation}"

    def test_imputation_zero(self):
        """
        the goal is to test if only the simple imputation with zero has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.import_matrix(np.array([[1, 2, 1], [4, 2, 6]]))

        contamination = np.array([[1, 2, np.nan], [4, np.nan, 6]])

        algo = Imputation.Statistics.ZeroImpute(contamination)
        algo.impute()
        algo.score(ts_1.data)

        result = np.array([[1, 2, 0], [4, 0, 6]])

        imputation, metrics = algo.imputed_matrix, algo.metrics

        assert np.all(np.isclose(imputation, result)), f"imputation: expected {result}, got {imputation}"

    def test_imputation_mean(self):
        """
        the goal is to test if only the simple imputation with mean has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.import_matrix(np.array([[4, 2, 1], [4, 2, 6]]))

        contamination = np.array([[4, 2, np.nan], [4, np.nan, 6]])

        algo = Imputation.Statistics.MeanImpute(contamination)
        algo.impute()
        algo.score(ts_1.data)

        result = np.array([[4, 2, 4], [4, 4, 6]])

        imputation, metrics = algo.imputed_matrix, algo.metrics

        assert np.all(np.isclose(imputation, result)), f"imputation: expected {result}, got {imputation}"
