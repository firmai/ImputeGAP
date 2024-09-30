import re
import os
import toml
from imputegap.algorithms.cdrec import cdrec
from imputegap.algorithms.iim import iim
from imputegap.algorithms.min_impute import min_impute
from imputegap.algorithms.mrnn import mrnn
from imputegap.algorithms.stmvl import stmvl
from imputegap.algorithms.zero_impute import zero_impute
from imputegap.evaluation.evaluation import Evaluation
from imputegap.manager import utils


class Imputation:


    def evaluate_params(ground_truth, contamination, configuration, algorithm="cdrec"):
        """
        evaluate various statistics for given parameters.
        @author : Quentin Nater

        :param ground_truth: original time series without contamination
        :param contamination: time series with contamination
        :param configuration : tuple of the configuration of the algorithm.
        :param algorithm : imputation algorithm to use | Valid values: 'cdrec', 'mrnn', 'stmvl', 'iim' | default = cdrec
        :param selected_metrics : list of selected metrics to compute | default = ["rmse"]
        :return: dict, a dictionary of computed statistics.
        """

        if algorithm == 'cdrec':
            rank, eps, iters = configuration
            imputation, error_measures = Imputation.MR.cdrec(ground_truth, contamination, (rank, eps, iters))
        elif algorithm == 'iim':
            learning_neighbours = configuration[0]
            alg_code = "iim " + re.sub(r'[\W_]', '', str(learning_neighbours))
            imputation, error_measures = Imputation.Regression.iim_imputation(ground_truth, contamination, (learning_neighbours, alg_code))
        elif algorithm == 'mrnn':
            hidden_dim, learning_rate, iterations = configuration
            imputation, error_measures = Imputation.ML.mrnn_imputation(ground_truth, contamination, (hidden_dim, learning_rate, iterations, 7))
        elif algorithm == 'stmvl':
            window_size, gamma, alpha = configuration
            imputation, error_measures = Imputation.Pattern.stmvl_imputation(ground_truth, contamination, (window_size, gamma, alpha))
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

        return error_measures

    class MR:
        def cdrec(ground_truth, contamination, params=None):
            """
            Imputation of data with CDREC algorithm
            @author Quentin Nater

            :param ground_truth: original time series without contamination
            :param contamination: time series with contamination
            :param params: [Optional] (rank, epsilon, iterations) : parameters of the algorithm, if None, default ones are loaded

            :return: imputed_matrix, metrics : all time series with imputation data and their metrics
            """
            if params is not None:
                rank, epsilon, iterations = params
            else:
                rank, epsilon, iterations = utils.load_parameters(query="default", algorithm="cdrec")

            imputed_matrix = cdrec(contamination=contamination, truncation_rank=rank, iterations=iterations, epsilon=epsilon)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            return imputed_matrix, metrics

    class Stats:
        def zero_impute(ground_truth, contamination, params=None):
            """
            Template zero impute for adding your own algorithms
            @author : Quentin Nater

            :param ground_truth: original time series without contamination
            :param contamination: time series with contamination
            :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

            :return: imputed_matrix, metrics : all time series with imputation data and their metrics
            """
            imputed_matrix = zero_impute(ground_truth, contamination, params)
            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            return imputed_matrix, metrics

        def min_impute(ground_truth, contamination, params=None):
            """
            Impute NaN values with the minimum value of the ground truth time series.
            @author : Quentin Nater

            :param ground_truth: original time series without contamination
            :param contamination: time series with contamination
            :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

            :return: imputed_matrix, metrics : all time series with imputation data and their metrics
            """
            imputed_matrix = min_impute(ground_truth, contamination, params)
            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            return imputed_matrix, metrics


    class Regression:
        def iim_imputation(ground_truth, contamination, params=None):
            """
           Imputation of data with IIM algorithm
           @author Quentin Nater

           :param ground_truth: original time series without contamination
           :param contamination: time series with contamination
           :param params: [Optional] (neighbors, algo_code) : parameters of the algorithm, if None, default ones are loaded : neighbors, algo_code

           :return: imputed_matrix, metrics : all time series with imputation data and their metrics
           """
            if params is not None:
                neighbors, algo_code = params
            else:
                neighbors, algo_code = utils.load_parameters(query="default", algorithm="iim")

            imputed_matrix = iim(contamination=contamination, number_neighbor=neighbors, algo_code=algo_code)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            return imputed_matrix, metrics

    class ML:
        def mrnn_imputation(ground_truth, contamination, params=None):
            """
           Imputation of data with MRNN algorithm
           @author Quentin Nater

           :param ground_truth: original time series without contamination
           :param contamination: time series with contamination
           :param params: [Optional] (hidden_dim, learning_rate, iterations, sequence_length) : parameters of the algorithm, hidden_dim, learning_rate, iterations, keep_prob, sequence_length, if None, default ones are loaded

           :return: imputed_matrix, metrics : all time series with imputation data and their metrics
           """
            if params is not None:
                hidden_dim, learning_rate, iterations, sequence_length = params
            else:
                hidden_dim, learning_rate, iterations, sequence_length = utils.load_parameters(query="default", algorithm="mrnn")

            imputed_matrix = mrnn(contamination=contamination, hidden_dim=hidden_dim, learning_rate=learning_rate, iterations=iterations, sequence_length=sequence_length)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            return imputed_matrix, metrics

    class Pattern:
        def stmvl_imputation(ground_truth, contamination, params=None):
            """
           Imputation of data with MRNN algorithm
           @author Quentin Nater

           :param ground_truth: original time series without contamination
           :param contamination: time series with contamination
           :param params: [Optional] (window_size, gamma, alpha) : parameters of the algorithm, window_size, gamma, alpha, if None, default ones are loaded
                :param window_size: window size for temporal component
                :param gamma: smoothing parameter for temporal weight
                :param alpha: power for spatial weight
           :return: imputed_matrix, metrics : all time series with imputation data and their metrics
           """
            if params is not None:
                window_size, gamma, alpha = params
            else:
                window_size, gamma, alpha = utils.load_parameters(query="default", algorithm="stmvl")

            imputed_matrix = stmvl(contamination=contamination, window_size=window_size, gamma=gamma, alpha=alpha)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            return imputed_matrix, metrics