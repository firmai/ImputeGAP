import re
from imputegap.algorithms.cdrec import cdrec
from imputegap.algorithms.iim import iim
from imputegap.algorithms.min_impute import min_impute
from imputegap.algorithms.mrnn import mrnn
from imputegap.algorithms.stmvl import stmvl
from imputegap.algorithms.zero_impute import zero_impute
from imputegap.tools.evaluation import Evaluation
from imputegap.tools import utils


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
            rank, epsilon, iterations = configuration
            algo = Imputation.MD.CDREC(contamination)
            algo.impute((rank, epsilon, iterations))
            algo.score(ground_truth, algo.imputed_matrix)
            imputation, error_measures = algo.imputed_matrix, algo.metrics
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

    class MD:

        class CDREC:

            def __init__(self, infected_matrix):
                """
                Store the results of the CDREC algorithm
                :param infected_matrix : Matrix used during the imputation of the time series
                """
                self.infected_matrix = infected_matrix
                self.imputed_matrix = None
                self.metrics = None
                self.optimal_params = None

            def impute(self, params=None):
                """
                Imputation of data with CDREC algorithm
                @author Quentin Nater

                :param params: [Optional] (rank, epsilon, iterations) : parameters of the algorithm, if None, default ones are loaded
                """
                if params is not None:
                    if isinstance(params, dict):
                        params = tuple(params.values())

                    rank, epsilon, iterations = params
                else:
                    rank, epsilon, iterations = utils.load_parameters(query="default", algorithm="cdrec")

                self.imputed_matrix = cdrec(contamination=self.infected_matrix, truncation_rank=int(rank), iterations=int(iterations), epsilon=float(epsilon))

            def score(self, raw_matrix, imputed_matrix):
                """
                Imputation of data with CDREC algorithm
                @author Quentin Nater

                :param raw_matrix: original time series without contamination
                :param infected_matrix: time series with contamination
                :param imputed_matrix: time series with imputation
                """
                self.metrics = Evaluation(raw_matrix, imputed_matrix, self.infected_matrix).metrics_computation()

            def optimize(self, raw_data, optimizer="bayesian", selected_metrics=["RMSE"], n_calls=3, n_random_starts=50, acq_func='gp_hedge'):
                """
                Conduct the optimization of the hyperparameters.

                Parameters
                ----------
                :param raw_data : time series data set to optimize
                :param optimizer : Choose the actual optimizer. | default "bayesian"
                :param selected_metrics : list of selected metrics to consider for optimization. | default ["RMSE"]
                :param n_calls: bayesian parameters, number of calls to the objective function.
                :param n_random_starts: bayesian parameters, number of initial calls to the objective function, from random points.
                :param acq_func: bayesian parameters, function to minimize over the Gaussian prior (one of 'LCB', 'EI', 'PI', 'gp_hedge') | default gp_hedgedge

                :return : Tuple[dict, Union[Union[int, float, complex], Any]], the best parameters and their corresponding scores.
                """
                from imputegap.recovery.optimization import Optimization

                if optimizer == "bayesian":
                    optimal_params, _ = Optimization.Bayesian.bayesian_optimization(ground_truth=raw_data, contamination=self.infected_matrix, selected_metrics=selected_metrics, n_calls=n_calls, n_random_starts=n_random_starts, acq_func=acq_func)
                self.optimal_params = optimal_params

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
                if isinstance(params, dict):
                    params = tuple(params.values())

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
                if isinstance(params, dict):
                    params = tuple(params.values())

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
                if isinstance(params, dict):
                    params = tuple(params.values())

                window_size, gamma, alpha = params
            else:
                window_size, gamma, alpha = utils.load_parameters(query="default", algorithm="stmvl")

            imputed_matrix = stmvl(contamination=contamination, window_size=window_size, gamma=gamma, alpha=alpha)

            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            return imputed_matrix, metrics