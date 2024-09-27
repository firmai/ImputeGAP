import os
import unittest
import numpy as np

from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation
from imputegap.manager import utils
from imputegap.manager.manager import TimeSeries




class TestIIM(unittest.TestCase):

    def test_imputation_iim_chlorine(self):
        """
        the goal is to test if only the simple imputation with IIM has the expected outcome
        """
        impute_gap = TimeSeries(utils.get_file_path_dataset("chlorine"))

        ts_contaminated = Contamination.scenario_mcar(ts=impute_gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=10,
                                                      protection=0.1, use_seed=True, seed=42)

        imputation, metrics = Imputation.Regression.iim_imputation(impute_gap.ts, ts_contaminated)

        expected_metrics = {
            "RMSE": 0.1634600944317234,
            "MAE": 0.12404674986835251,
            "MI": 0.6446689074332342,
            "CORRELATION": 0.8788051903986334
        }

        impute_gap.ts_contaminate = ts_contaminated
        impute_gap.ts_imputation = imputation
        impute_gap.metrics = metrics
        impute_gap.print_results()

        assert np.isclose(round(metrics["RMSE"], 3), round(expected_metrics["RMSE"], 3), atol=0.01)
        assert np.isclose(round(metrics["MAE"], 3), round(expected_metrics["MAE"], 3), atol=0.01)
        assert np.isclose(round(metrics["MI"], 3), round(expected_metrics["MI"], 3), atol=0.01)
        assert np.isclose(round(metrics["CORRELATION"], 3), round(expected_metrics["CORRELATION"], 3), atol=0.01)