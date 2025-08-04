import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestGROUSE(unittest.TestCase):

    def test_imputation_grouse_dft(self):
        """
        the goal is to test if only the simple imputation with GROUSE has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.GROUSE(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 0.18874406368396715,
            "MAE": 0.1510815475102299,
            "MI": 0.12630593106068652,
            "CORRELATION": 0.4450326583619584
        }

        assert np.isclose(metrics["RMSE"], expected_metrics["RMSE"]), f"RMSE mismatch: expected {expected_metrics['RMSE']}, got {metrics['RMSE']}"
        assert np.isclose(metrics["MAE"], expected_metrics["MAE"]), f"MAE mismatch: expected {expected_metrics['MAE']}, got {metrics['MAE']}"
        assert np.isclose(metrics["MI"], expected_metrics["MI"]), f"MI mismatch: expected {expected_metrics['MI']}, got {metrics['MI']}"
        assert np.isclose(metrics["CORRELATION"], expected_metrics["CORRELATION"]), f"Correlation mismatch: expected {expected_metrics['CORRELATION']}, got {metrics['CORRELATION']}"

    def test_imputation_grouse_udef(self):
        """
        the goal is to test if only the simple imputation with GROUSE has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.GROUSE(incomp_data).impute(params={"max_rank": 5})
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 0.1917033510202887,
            "MAE": 0.14916266066759773,
            "MI": 0.12794381368232188,
            "CORRELATION": 0.43135391546458096
        }

        assert np.isclose(metrics["RMSE"], expected_metrics["RMSE"]), f"RMSE mismatch: expected {expected_metrics['RMSE']}, got {metrics['RMSE']}"
        assert np.isclose(metrics["MAE"], expected_metrics["MAE"]), f"MAE mismatch: expected {expected_metrics['MAE']}, got {metrics['MAE']}"
        assert np.isclose(metrics["MI"], expected_metrics["MI"]), f"MI mismatch: expected {expected_metrics['MI']}, got {metrics['MI']}"
        assert np.isclose(metrics["CORRELATION"], expected_metrics["CORRELATION"]), f"Correlation mismatch: expected {expected_metrics['CORRELATION']}, got {metrics['CORRELATION']}"