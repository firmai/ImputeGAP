import unittest
import numpy as np

from imputegap.recovery.contamination import Contamination
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestMRNN(unittest.TestCase):

    def test_imputation_mrnn_chlorine(self):
        """
        the goal is to test if only the simple imputation with MRNN has the expected outcome
        """
        impute_gap = TimeSeries(utils.get_file_path_dataset("chlorine"))

        ts_contaminated = Contamination.scenario_mcar(ts=impute_gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=10,
                                                      protection=0.1, use_seed=True, seed=42)

        imputation, metrics = Imputation.ML.mrnn_imputation(impute_gap.ts, ts_contaminated)

        expected_metrics = {
            "RMSE": 0.34949060530462533,
            "MAE": 0.25223314039410877,
            "MI": 0.0013520391456194952,
            "CORRELATION": 0.11633073201054478
        }

        impute_gap.ts_contaminate = ts_contaminated
        impute_gap.ts_imputation = imputation
        impute_gap.metrics = metrics
        impute_gap.print_results()

        assert np.isclose(round(metrics["RMSE"], 3), round(expected_metrics["RMSE"], 3), atol=0.01)
        assert np.isclose(round(metrics["MAE"], 3), round(expected_metrics["MAE"], 3), atol=0.01)
        assert np.isclose(round(metrics["MI"], 3), round(expected_metrics["MI"], 3), atol=0.01)
        assert np.isclose(round(metrics["CORRELATION"], 3), round(expected_metrics["CORRELATION"], 3), atol=0.01)