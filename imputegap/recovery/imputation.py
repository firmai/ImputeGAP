import re
from imputegap.algorithms.cdrec import cdrec
from imputegap.algorithms.iim import iim
from imputegap.algorithms.min_impute import min_impute
from imputegap.algorithms.mrnn import mrnn
from imputegap.algorithms.stmvl import stmvl
from imputegap.algorithms.zero_impute import zero_impute
from imputegap.tools.evaluation import Evaluation
from imputegap.tools import utils


class BaseImputer:
    algorithm = None  # Class variable to hold the algorithm name

    def __init__(self, infected_matrix):
        """
        Store the results of the imputation algorithm.
        :param infected_matrix : Matrix used during the imputation of the time series
        """
        self.infected_matrix = infected_matrix
        self.imputed_matrix = None
        self.metrics = None
        self.parameters = None

    def impute(self, params=None):
        raise NotImplementedError("This method should be overridden by subclasses")

    def score(self, raw_matrix, imputed_matrix=None):
        """
        Imputation of data with CDREC algorithm
        @author Quentin Nater

        :param raw_matrix: original time series without contamination
        :param infected_matrix: time series with contamination
        """
        self.metrics = Evaluation(raw_matrix, self.imputed_matrix, self.infected_matrix).metrics_computation()

    def _check_params(self, params):
        """
        Format the parameters for optimization or imputation
        :param params: list or dictionary of parameters
        :return: tuples of parameters in the right format
        """
        if params is not None:
            if isinstance(params, dict):
                params = tuple(params.values())

            if 'automl' in params:
                print("\noptimizer has been called...\n")
                flag, raw_data, *rest = params
                self._optimize(raw_data, *rest)

                if isinstance(self.parameters, dict):
                    self.parameters = tuple(self.parameters.values())

            else:
                self.parameters = params

            if self.algorithm == "iim":
                if len(self.parameters) == 1:
                    learning_neighbours = self.parameters[0]
                    algo_code = "iim " + re.sub(r'[\W_]', '', str(learning_neighbours))
                    self.parameters = (learning_neighbours, algo_code)

            if self.algorithm == "mrnn":
                if len(self.parameters) == 3:
                    hidden_dim, learning_rate, iterations = self.parameters
                    _, _, _, sequence_length = utils.load_parameters(query="default", algorithm="mrnn")
                    self.parameters = (hidden_dim, learning_rate, iterations, sequence_length)

        return self.parameters

    def _optimize(self, raw_data, optimizer="bayesian", n_calls=3, metrics=["RMSE"], random_starts=50, func='gp_hedge'):
        """
        Conduct the optimization of the hyperparameters.

        Parameters
        ----------
        :param raw_data : time series data set to optimize
        :param optimizer : Choose the actual optimizer. | default "bayesian"
        :param metrics : list of selected metrics to consider for optimization. | default ["RMSE"]
        :param n_calls: bayesian parameters, number of calls to the objective function.
        :param random_starts: bayesian parameters, number of initial calls to the objective function, from random points.
        :param func: bayesian parameters, function to minimize over the Gaussian prior (one of 'LCB', 'EI', 'PI', 'gp_hedge') | default gp_hedgedge

        :return : Tuple[dict, Union[Union[int, float, complex], Any]], the best parameters and their corresponding scores.
        """
        from imputegap.recovery.optimization import Optimization

        if optimizer == "bayesian":
            optimal_params, _ = Optimization.Bayesian.optimize(ground_truth=raw_data,
                                                               contamination=self.infected_matrix,
                                                               selected_metrics=metrics,
                                                               algorithm=self.algorithm,
                                                               n_calls=n_calls,
                                                               n_random_starts=random_starts,
                                                               acq_func=func)
        elif optimizer == "greedy":
            optimal_params, _ = Optimization.Greedy.optimize(ground_truth=raw_data,
                                                             contamination=self.infected_matrix,
                                                             selected_metrics=metrics, algorithm=self.algorithm,
                                                             n_calls=250)

        self.parameters = optimal_params


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

        if isinstance(configuration, dict):
            configuration = tuple(configuration.values())

        if algorithm == 'cdrec':
            rank, epsilon, iterations = configuration
            algo = Imputation.MD.CDRec(contamination)
            algo.impute((rank, epsilon, iterations))

        elif algorithm == 'iim':
            learning_neighbours = configuration[0]
            alg_code = "iim " + re.sub(r'[\W_]', '', str(learning_neighbours))

            algo = Imputation.Regression.IIM(contamination)
            algo.impute((learning_neighbours, alg_code))

        elif algorithm == 'mrnn':
            hidden_dim, learning_rate, iterations = configuration

            algo = Imputation.ML.MRNN(contamination)
            algo.impute((hidden_dim, learning_rate, iterations, 7))

        elif algorithm == 'stmvl':
            window_size, gamma, alpha = configuration

            algo = Imputation.Pattern.STMVL(contamination)
            algo.impute((window_size, gamma, alpha))

        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

        algo.score(ground_truth)
        imputation, error_measures = algo.imputed_matrix, algo.metrics

        return error_measures

    class Stats:

        class ZeroImpute(BaseImputer):
            algorithm = "zero_impute"

            def impute(self, params=None):
                """
                Template zero impute for adding your own algorithms
                @author : Quentin Nater

                :param ground_truth: original time series without contamination
                :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

                :return: imputed_matrix, metrics : all time series with imputation data and their metrics
                """
                self.imputed_matrix = zero_impute(self.infected_matrix, params)

                return self

        class MinImpute(BaseImputer):
            algorithm = "min_impute"

            def impute(self, params=None):
                """
                Impute NaN values with the minimum value of the ground truth time series.
                @author : Quentin Nater

                :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

                :return: imputed_matrix, metrics : all time series with imputation data and their metrics
                """
                self.imputed_matrix = min_impute(self.infected_matrix, params)

                return self

    class MD:
        class CDRec(BaseImputer):
            algorithm = "cdrec"

            def impute(self, params=None):
                """
                Imputation of data with CDREC algorithm
                @author Quentin Nater

                :param params: [Optional-IMPUTATION] parameters of the algorithm, if None, default ones are loaded
                               [Optional-AUTO_ML]  parameters of the automl, if None, default ones are loaded

                option 1 : algorithm parameters ___________________________________________________

                    tuples((int) truncation_rank, (float) epsilon, (int) iterations)

                    truncation_rank: rank of reduction of the matrix (must be higher than 1 and smaller than the limit of series)

                    epsilon : learning rate

                    iterations : number of iterations


                option 2 : automl parameters________________________________________________________

                    tuples((str) flag, (numpy) ground truth, (str) optimizer*, (int) n_calls*, List(str) selected_metrics*, (int) n_random_starts*, (str) acq_func*)

                    flag : activate or not the optimization : "automl"

                    numpy : ground truth, TimeSeries().data

                    optimizer : [OPTIONAL] choice of the optimizer : "bayesian" or "greedy"  | default "bayesian"

                    n_calls: [OPTIONAL] bayesian parameters, number of calls to the objective function. | default 3

                    selected_metrics : [OPTIONAL] list of selected metrics to consider for optimization. | default ["RMSE"]

                    n_random_starts: [OPTIONAL] bayesian parameters, number of initial calls to the objective function, from random points. | default 50

                    acq_func: [OPTIONAL] bayesian parameters, function to minimize over the Gaussian prior (one of 'LCB', 'EI', 'PI', 'gp_hedge') | default gp_hedge

                """
                if params is not None:
                    rank, epsilon, iterations = self._check_params(params)
                else:
                    rank, epsilon, iterations = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.imputed_matrix = cdrec(contamination=self.infected_matrix, truncation_rank=rank, iterations=iterations, epsilon=epsilon)

                return self

    class Regression:

        class IIM(BaseImputer):
            algorithm = "iim"

            def impute(self, params=None):
                """
               Imputation of data with IIM algorithm
               @author Quentin Nater

               :param params: [Optional] (neighbors, algo_code) : parameters of the algorithm, if None, default ones are loaded : neighbors, algo_code

               :return: imputed_matrix, metrics : all time series with imputation data and their metrics
               """
                if params is not None:
                    learning_neighbours, algo_code = self._check_params(params)
                else:
                    learning_neighbours, algo_code = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.imputed_matrix = iim(contamination=self.infected_matrix, number_neighbor=learning_neighbours, algo_code=algo_code)

                return self

    class ML:
        class MRNN(BaseImputer):
            algorithm = "mrnn"

            def impute(self, params=None):
                """
               Imputation of data with MRNN algorithm
               @author Quentin Nater

               :param params: [Optional] (hidden_dim, learning_rate, iterations, sequence_length) : parameters of the algorithm, hidden_dim, learning_rate, iterations, keep_prob, sequence_length, if None, default ones are loaded

               :return: imputed_matrix, metrics : all time series with imputation data and their metrics
               """
                if params is not None:
                    hidden_dim, learning_rate, iterations, sequence_length = self._check_params(params)
                else:
                    hidden_dim, learning_rate, iterations, sequence_length = utils.load_parameters(query="default", algorithm="mrnn")

                self.imputed_matrix = mrnn(contamination=self.infected_matrix, hidden_dim=hidden_dim,
                                           learning_rate=learning_rate, iterations=iterations,
                                           sequence_length=sequence_length)

                return self

    class Pattern:

        class STMVL(BaseImputer):
            algorithm = "stmvl"

            def impute(self, params=None):
                """
               Imputation of data with MRNN algorithm
               @author Quentin Nater

               :param params: [Optional] (window_size, gamma, alpha) : parameters of the algorithm, window_size, gamma, alpha, if None, default ones are loaded
                    :param window_size: window size for temporal component
                    :param gamma: smoothing parameter for temporal weight
                    :param alpha: power for spatial weight
               :return: imputed_matrix, metrics : all time series with imputation data and their metrics
               """
                if params is not None:
                    window_size, gamma, alpha = params = self._check_params(params)
                else:
                    window_size, gamma, alpha = utils.load_parameters(query="default", algorithm="stmvl")

                self.imputed_matrix = stmvl(contamination=self.infected_matrix, window_size=window_size, gamma=gamma, alpha=alpha)

                return self
