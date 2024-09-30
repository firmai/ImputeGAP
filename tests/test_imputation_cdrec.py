import os
import unittest
import numpy as np

from imputegap.recovery.contamination import Contamination
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


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


class TestCDREC(unittest.TestCase):

    def test_imputation_cdrec(self):
        """
        the goal is to test if only the simple imputation with cdrec has the expected outcome
        """
        impute_gap = TimeSeries((utils.get_file_path_dataset("test")))

        ts_contaminated = Contamination.scenario_mcar(ts=impute_gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=2, protection=0.1, use_seed=True, seed=42)
        imputation, metrics = Imputation.MR.cdrec(impute_gap.ts, ts_contaminated)

        #assert not np.isnan(imputation).any(), "The imputed data contains NaN values."

        expected_metrics = {
            "RMSE": 0.5993259196563864,
            "MAE": 0.5190054811092809,
            "MI": 0.7796564257088495,
            "CORRELATION": 0.6358270633906415
        }

        impute_gap.ts_contaminate = ts_contaminated
        impute_gap.ts_imputation = imputation
        impute_gap.metrics = metrics
        impute_gap.print_results()

        assert np.isclose(metrics["RMSE"], expected_metrics["RMSE"]), f"RMSE mismatch: expected {expected_metrics['RMSE']}, got {metrics['RMSE']}"
        assert np.isclose(metrics["MAE"], expected_metrics["MAE"]), f"MAE mismatch: expected {expected_metrics['MAE']}, got {metrics['MAE']}"
        assert np.isclose(metrics["MI"], expected_metrics["MI"]), f"MI mismatch: expected {expected_metrics['MI']}, got {metrics['MI']}"
        assert np.isclose(metrics["CORRELATION"], expected_metrics["CORRELATION"]), f"Correlation mismatch: expected {expected_metrics['CORRELATION']}, got {metrics['CORRELATION']}"

    def test_imputation_cdrec_chlorine(self):
        """
        the goal is to test if only the simple imputation with cdrec has the expected outcome
        """
        impute_gap = TimeSeries(utils.get_file_path_dataset("chlorine"), limitation_values=200)

        ts_contaminated = Contamination.scenario_mcar(ts=impute_gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=10,
                                                      protection=0.1,
                                                      use_seed=True, seed=42)

        imputation, metrics = Imputation.MR.cdrec(impute_gap.ts, ts_contaminated)

        # assert not np.isnan(imputation).any(), "The imputed data contains NaN values."

        expected_metrics = {
            "RMSE": 0.10329523970909142,
            "MAE": 0.06717112854576478,
            "MI": 0.7706445457837339,
            "CORRELATION": 0.913368365805848
        }

        impute_gap.ts_contaminate = ts_contaminated
        impute_gap.ts_imputation = imputation
        impute_gap.metrics = metrics
        impute_gap.print_results()

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < 0.1, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < 0.1, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")
        self.assertTrue(abs(metrics["MI"] - expected_metrics["MI"]) < 0.1, f"metrics MI = {metrics['MI']}, expected MI = {expected_metrics['MI']} ")
        self.assertTrue(abs(metrics["CORRELATION"] - expected_metrics["CORRELATION"]) < 0.1, f"metrics CORRELATION = {metrics['CORRELATION']}, expected CORRELATION = {expected_metrics['CORRELATION']} ")
