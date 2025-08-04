import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestSVT(unittest.TestCase):

    def test_imputation_svt_dft(self):
        """
        the goal is to test if only the simple imputation with SVT has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.SVT(incomp_data).impute()
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 0.05559479346635801,
            "MAE": 0.04122525697280994,
            "MI": 1.1183928744751146,
            "CORRELATION": 0.9671521894143095
        }

        assert np.isclose(metrics["RMSE"], expected_metrics["RMSE"]), f"RMSE mismatch: expected {expected_metrics['RMSE']}, got {metrics['RMSE']}"
        assert np.isclose(metrics["MAE"], expected_metrics["MAE"]), f"MAE mismatch: expected {expected_metrics['MAE']}, got {metrics['MAE']}"
        assert np.isclose(metrics["MI"], expected_metrics["MI"]), f"MI mismatch: expected {expected_metrics['MI']}, got {metrics['MI']}"
        assert np.isclose(metrics["CORRELATION"], expected_metrics["CORRELATION"]), f"Correlation mismatch: expected {expected_metrics['CORRELATION']}, got {metrics['CORRELATION']}"

    def test_imputation_svt_udef(self):
        """
        the goal is to test if only the simple imputation with SVT has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.SVT(incomp_data).impute(params={"tau": 0.5})
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 0.07170536160702093,
            "MAE": 0.05510518767210485,
            "MI": 0.9566331635748319,
            "CORRELATION": 0.9429733660164527
        }

        assert np.isclose(metrics["RMSE"], expected_metrics["RMSE"]), f"RMSE mismatch: expected {expected_metrics['RMSE']}, got {metrics['RMSE']}"
        assert np.isclose(metrics["MAE"], expected_metrics["MAE"]), f"MAE mismatch: expected {expected_metrics['MAE']}, got {metrics['MAE']}"
        assert np.isclose(metrics["MI"], expected_metrics["MI"]), f"MI mismatch: expected {expected_metrics['MI']}, got {metrics['MI']}"
        assert np.isclose(metrics["CORRELATION"], expected_metrics["CORRELATION"]), f"Correlation mismatch: expected {expected_metrics['CORRELATION']}, got {metrics['CORRELATION']}"