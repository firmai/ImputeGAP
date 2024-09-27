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


class Imputation:

    def load_parameters(query="default", algorithm="cdrec"):
        """
        Load default values of algorithms

        :param query : ('optimal' or 'default'), load default or optimal parameters for algorithms | default "default"
        :param algorithm : algorithm parameters to load | default "cdrec"

        :return: tuples of optimal parameters and the config of default values
        """

        if query == "default":
            filepath = "../env/default_values.toml"
        elif query == "optimal":
            filepath = "../env/optimal_parameters_"+str(algorithm)+".toml"
        else:
            print("Query not found for this function ('optimal' or 'default')")

        if not os.path.exists(filepath):
            filepath = filepath[:1]

        with open(filepath, "r") as file:
            config = toml.load(filepath)

        params = None
        if algorithm == "cdrec":
            truncation_rank = int(config['cdrec']['rank'])
            epsilon = config['cdrec']['epsilon']
            iterations = int(config['cdrec']['iteration'])
            params = (truncation_rank, epsilon, iterations)
        elif algorithm == "stmvl":
            window_size = int(config['stmvl']['window_size'])
            gamma = float(config['stmvl']['gamma'])
            alpha = int(config['stmvl']['alpha'])
            params = (window_size, gamma, alpha)
        elif algorithm == "iim":
            neighbors = int(config['iim']['neighbor'])
            algo_code = config['iim']['algorithm_code']
            params = (neighbors, algo_code)
        elif algorithm == "mrnn":
            hidden_dim = int(config['mrnn']['hidden_dim'])
            learning_rate = float(config['mrnn']['learning_rate'])
            iterations = int(config['mrnn']['iterations'])
            sequence_length = int(config['mrnn']['sequence_length'])
            params = (hidden_dim, learning_rate, iterations, sequence_length)
        else :
            print("Default/Optimal config not found for this algorithm")

        return params

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
            learning_neighbours = configuration
            alg_code = "iim " + re.sub(r'[\W_]', '', str(learning_neighbours))
            imputation, error_measures = Imputation.Regression.iim_imputation(ground_truth, contamination, (learning_neighbours, alg_code))
        elif algorithm == 'mrnn':
            hidden_dim, learning_rate, iterations, keep_prob, seq_len = configuration
            imputation, error_measures = Imputation.ML.mrnn_imputation(ground_truth, contamination, (hidden_dim, learning_rate, iterations, seq_len))
        elif algorithm == 'stmvl':
            window_size, gamma, alpha = configuration
            imputation, error_measures = Imputation.Pattern.stmvl_imputation(ground_truth, contamination, (window_size, gamma, alpha))
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

        #print("error_measures :", error_measures)

        return error_measures

    class MR:
        def cdrec(ground_truth, contamination, params=None):
            """
            Imputation of data with CDREC algorithm
            @author Quentin Nater

            :param ground_truth: original time series without contamination
            :param contamination: time series with contamination
            :param params: [Optional] (truncation_rank, epsilon, iterations) : parameters of the algorithm, if None, default ones are loaded

            :return: imputed_matrix, metrics : all time series with imputation data and their metrics
            """
            if params is not None:
                truncation_rank, epsilon, iterations = params
            else:
                truncation_rank, epsilon, iterations = Imputation.load_parameters(algorithm="cdrec")

            imputed_matrix = cdrec(contamination=contamination, truncation_rank=truncation_rank, iterations=iterations, epsilon=epsilon)

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
                neighbors, algo_code = Imputation.load_parameters(algorithm="iim")

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
                hidden_dim, learning_rate, iterations, sequence_length = Imputation.load_parameters(algorithm="mrnn")

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
                window_size, gamma, alpha = Imputation.load_parameters(algorithm="stmvl")

            imputed_matrix = stmvl(contamination=contamination, window_size=window_size, gamma=gamma, alpha=alpha)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            return imputed_matrix, metrics