import os
import toml
from imputegap.algorithms.cdrec import cdrec
from imputegap.algorithms.iim import iim
from imputegap.algorithms.min_impute import min_impute
from imputegap.algorithms.mrnn import mrnn
from imputegap.algorithms.stmvl import stmvl
from imputegap.algorithms.zero_impute import zero_impute
from imputegap.evaluation.evaluation import Evaluation


class Imputation:

    def load_toml(filepath = "../env/default_values.toml"):
        """
        Load default values of algorithms
        :return: the config of default values
        """
        if not os.path.exists(filepath):
            filepath = "./env/default_values.toml"

        with open(filepath, "r") as file:
            config = toml.load(file)

        return config

    class MR:
        def cdrec(ground_truth, contamination, params=None, display=False):
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
                config = Imputation.load_toml()
                truncation_rank = config['cdrec']['default_reduction_rank']
                epsilon = config['cdrec']['default_epsilon']
                iterations = config['cdrec']['default_iteration']
                params = truncation_rank, epsilon, iterations

            print("\n\tCDREC Imputation lanched with : ", params)

            imputed_matrix = cdrec(contamination=contamination, truncation_rank=truncation_rank, iterations=iterations, epsilon=epsilon)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            print("\n\tCDREC Imputation completed without error.\n")

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

            print("\n\n\tZERO Imputation completed without error.\n")

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

            print("\n\n\tMIN Imputation completed without error.\n")

            return imputed_matrix, metrics


    class Regression:
        def iim_imputation(ground_truth, contamination, params=None):
            """
           Imputation of data with IIM algorithm
           @author Quentin Nater

           :param ground_truth: original time series without contamination
           :param contamination: time series with contamination
           :param params: [Optional] parameters of the algorithm, if None, default ones are loaded : neighbors, algo_code

           :return: imputed_matrix, metrics : all time series with imputation data and their metrics
           """
            if params is not None:
                neighbors, algo_code = params
            else:
                config = Imputation.load_toml()
                neighbors = config['iim']['default_neighbor']
                algo_code = config['iim']['default_algorithm_code']

            print("\n\n\tIIM Imputation lanched...\n")

            imputed_matrix = iim(contamination=contamination, number_neighbor=neighbors, algo_code=algo_code)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            print("\n\t\tIIM Imputation completed without error.\n")

            return imputed_matrix, metrics

    class ML:
        def mrnn_imputation(ground_truth, contamination, params=None):
            """
           Imputation of data with MRNN algorithm
           @author Quentin Nater

           :param ground_truth: original time series without contamination
           :param contamination: time series with contamination
           :param params: [Optional] parameters of the algorithm, hidden_dim, learning_rate, iterations, keep_prob, sequence_length, if None, default ones are loaded

           :return: imputed_matrix, metrics : all time series with imputation data and their metrics
           """
            if params is not None:
                hidden_dim, learning_rate, iterations, sequence_length = params
            else:
                config = Imputation.load_toml()
                hidden_dim = config['mrnn']['default_hidden_dim']
                learning_rate = config['mrnn']['default_learning_rate']
                iterations = config['mrnn']['default_iterations']
                sequence_length = config['mrnn']['default_sequence_length']

            print("\n\n\tMRNN Imputation lanched...\n")

            imputed_matrix = mrnn(contamination=contamination, hidden_dim=hidden_dim, learning_rate=learning_rate, iterations=iterations, sequence_length=sequence_length)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            print("\n\tMRNN Imputation completed without error.\n")

            return imputed_matrix, metrics

    class Pattern:
        def stmvl_imputation(ground_truth, contamination, params=None):
            """
           Imputation of data with MRNN algorithm
           @author Quentin Nater

           :param ground_truth: original time series without contamination
           :param contamination: time series with contamination
           :param params: [Optional] parameters of the algorithm, window_size, gamma, alpha, if None, default ones are loaded
                :param window_size: window size for temporal component
                :param gamma: smoothing parameter for temporal weight
                :param alpha: power for spatial weight
           :return: imputed_matrix, metrics : all time series with imputation data and their metrics
           """
            if params is not None:
                window_size, gamma, alpha = params
            else:
                config = Imputation.load_toml()
                window_size = config['stmvl']['default_window_size']
                gamma = config['stmvl']['default_gamma']
                alpha = config['stmvl']['default_alpha']

            print("\n\n\tST-MVL Imputation lanched...\n")

            imputed_matrix = stmvl(contamination=contamination, window_size=window_size, gamma=gamma, alpha=alpha)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            print("\n\tST-MVL Imputation completed without error.\n")

            return imputed_matrix, metrics