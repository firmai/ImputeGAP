import unittest
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestSPIRIT(unittest.TestCase):

    def test_imputation_spirit_dft(self):
        """
        the goal is to test if only the simple imputation with SPIRIT has the expected outcome
        """
        # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
        ts_x = TimeSeries()
        ts_x.load_series(utils.search_path("airq"))
        ts_x.data = ts_x.data.T

        miss_ts = ts_x.Contamination.aligned(ts_x.data.T, rate_dataset=0.1, rate_series=0.18)
        miss_ts = miss_ts.T

        algo2 = Imputation.MatrixCompletion.SPIRIT(miss_ts).impute()
        algo2.score(ts_x.data, algo2.recov_data)
        metrics = algo2.metrics
        ts_x.print_results(algo2.metrics, algorithm=algo2.algorithm)

        expected_metrics = {
            "RMSE": 0.38937914191833917,
            "MAE": 0.25818866227956544,
            "MI": 1.1257311598908373,
            "CORRELATION": 0.9383991348665752
        }

        assert np.isclose(metrics["RMSE"], expected_metrics["RMSE"]), f"RMSE mismatch: expected {expected_metrics['RMSE']}, got {metrics['RMSE']}"
        assert np.isclose(metrics["MAE"], expected_metrics["MAE"]), f"MAE mismatch: expected {expected_metrics['MAE']}, got {metrics['MAE']}"
        assert np.isclose(metrics["MI"], expected_metrics["MI"]), f"MI mismatch: expected {expected_metrics['MI']}, got {metrics['MI']}"
        assert np.isclose(metrics["CORRELATION"], expected_metrics["CORRELATION"]), f"Correlation mismatch: expected {expected_metrics['CORRELATION']}, got {metrics['CORRELATION']}"

    def test_imputation_spirit_udef(self):
        """
        the goal is to test if only the simple imputation with SPIRIT has the expected outcome
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("eeg-alcohol"))
        ts_1.normalize(normalizer="min_max")

        incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=10, offset=0.1, seed=True)

        algo = Imputation.MatrixCompletion.SPIRIT(incomp_data).impute(params={"k": 3, "w": 3, "lvalue": 0.5})
        algo.score(ts_1.data)
        metrics = algo.metrics

        expected_metrics = {
            "RMSE": 100,
            "MAE": 100,
            "MI": 0,
            "CORRELATION": np.nan
        }

        assert np.isclose(metrics["RMSE"], expected_metrics["RMSE"]), f"RMSE mismatch: expected {expected_metrics['RMSE']}, got {metrics['RMSE']}"
        assert np.isclose(metrics["MAE"], expected_metrics["MAE"]), f"MAE mismatch: expected {expected_metrics['MAE']}, got {metrics['MAE']}"
        assert np.isclose(metrics["MI"], expected_metrics["MI"]), f"MI mismatch: expected {expected_metrics['MI']}, got {metrics['MI']}"