import os

import toml
from imputegap.algorithms.cdrec import native_cdrec_param
from imputegap.evaluation._evaluation import EvaluationGAP


class ImputationGAP:

    def __init__(self):
        """
        Initialize the ImputationGAP class.

        :param gap: the TimeSeriesGAP object managing the time series.
        """
        self.config = self.load_toml()

    def load_toml(self, filepath = "../env/default_values.toml"):
        """
        Load default values of algorithms
        :return: the config of default values
        """
        if not os.path.exists(filepath):
            filepath = "./env/default_values.toml"

        with open(filepath, "r") as file:
            config = toml.load(file)
        return config

    def metrics_computation(self, ground_truth, imputation, contamination):
        """
        Compute the metrics to express the results of the imputation based on the ground truth and the contamination set

        :param ground_truth: original time series without contamination
        :param imputation: new time series with imputation values
        :param contamination: time series with contamination
        :return: metrics, dictionary containing each metric of the imputation
        """
        evaluation = EvaluationGAP(ground_truth, imputation, contamination)  # test, to change

        rmse = evaluation.compute_rmse()
        mae = evaluation.compute_mae()
        mi_d = evaluation.compute_mi()
        correlation = evaluation.compute_correlation()

        metrics = {"RMSE": rmse, "MAE": mae, "MI": mi_d, "CORRELATION": correlation}

        return metrics

    def cdrec(self, ground_truth, contamination, params=None):
        """
        Imputation of data with CDREC algorithm
        @author Quentin Nater

        :param ground_truth: original time series without contamination
        :param contamination: time series with contamination
        :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

        :return: imputed_matrix, metrics : all time series with imputation data and their metrics
        """

        if params is not None:
            truncation_rank, epsilon, iterations = params
        else:
            truncation_rank = self.config['cdrec']['default_reduction_rank']
            epsilon = self.config['cdrec']['default_epsilon_str']
            iterations = self.config['cdrec']['default_iteration']

        imputed_matrix = native_cdrec_param(__py_matrix=contamination, __py_rank=int(truncation_rank),
                                            __py_eps=float("1" + epsilon), __py_iters=int(iterations))

        metrics = self.metrics_computation(ground_truth, ground_truth+0.1, contamination)

        print("CDREC Imputation completed without error.\n")

        return imputed_matrix, metrics
