import os
import unittest
import numpy as np

from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation
from imputegap.manager.manager import TimeSeries


def resolve_path(local_path, github_actions_path):
    """
    Find the accurate path for tests

    :param local_path: path of local code
    :param github_actions_path: path on GitHub action
    :return: correct file paths
    """
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(github_actions_path):
        return github_actions_path
    else:
        raise FileNotFoundError("File not found in both: ", local_path, " and ", github_actions_path)


def get_save_path():
    """
    Find the accurate path for saving files of tests
    :return: correct file paths
    """
    return resolve_path('../tests/assets', './tests/assets')


def get_file_path(set_name="test"):
    """
    Find the accurate path for loading files of tests
    :return: correct file paths
    """
    return resolve_path(f'../imputegap/dataset/{set_name}.txt', f'./imputegap/dataset/{set_name}.txt')


class TestContamination(unittest.TestCase):

    def test_imputation_iim_chlorine(self):
        """
        the goal is to test if only the simple imputation with cdrec has the expected outcome
        """
        impute_gap = TimeSeries(get_file_path("chlorine"))

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

        assert np.isclose(round(metrics["RMSE"], 3), round(expected_metrics["RMSE"], 3))
        assert np.isclose(round(metrics["MAE"], 3), round(expected_metrics["MAE"], 3))
        assert np.isclose(round(metrics["MI"], 3), round(expected_metrics["MI"], 3))
        assert np.isclose(round(metrics["CORRELATION"], 3), round(expected_metrics["CORRELATION"], 3))