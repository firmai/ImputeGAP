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
    logs = True

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
        :author: Quentin Nater

        :param raw_matrix: original time series without contamination
        :param infected_matrix: time series with contamination
        """
        if self.imputed_matrix is None:
            self.imputed_matrix = imputed_matrix

        self.metrics = Evaluation(raw_matrix, self.imputed_matrix, self.infected_matrix).metrics_computation()

    def _check_params(self, user_defined, params):
        """
        Format the parameters for optimization or imputation
        :param params: list or dictionary of parameters
        :return: tuples of parameters in the right format
        """

        if params is not None:
            if not user_defined:
                self._optimize(params)

                if isinstance(self.parameters, dict):
                    self.parameters = tuple(self.parameters.values())

            else:
                if isinstance(params, dict):
                    params = tuple(params.values())

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

    def _optimize(self, parameters={}):
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

        raw_data = parameters.get('ground_truth')
        if raw_data is None:
            raise ValueError(f"Need ground_truth to be able to adapt the hyper-parameters: {raw_data}")

        optimizer = parameters.get('optimizer', "bayesian")
        defaults = utils.load_parameters(query="default", algorithm=optimizer)

        print("\noptimizer", optimizer, "has been called with", self.algorithm, "...\n")

        if optimizer == "bayesian":
            n_calls_d, n_random_starts_d, acq_func_d, selected_metrics_d = defaults
            options = parameters.get('options', {})

            n_calls = options.get('n_calls', n_calls_d)
            random_starts = options.get('n_random_starts', n_random_starts_d)
            func = options.get('acq_func', acq_func_d)
            metrics = options.get('selected_metrics', selected_metrics_d)

            optimal_params, _ = Optimization.Bayesian.optimize(ground_truth=raw_data,
                                                               contamination=self.infected_matrix,
                                                               selected_metrics=metrics,
                                                               algorithm=self.algorithm,
                                                               n_calls=n_calls,
                                                               n_random_starts=random_starts,
                                                               acq_func=func)
        elif optimizer == "pso":

            n_particles_d, c1_d, c2_d, w_d, iterations_d, n_processes_d, selected_metrics_d = defaults
            options = parameters.get('options', {})

            n_particles = options.get('n_particles', n_particles_d)
            c1 = options.get('c1', c1_d)
            c2 = options.get('c2', c2_d)
            w = options.get('w', w_d)
            iterations = options.get('iterations', iterations_d)
            n_processes = options.get('n_processes', n_processes_d)
            metrics = options.get('selected_metrics', selected_metrics_d)

            swarm_optimizer = Optimization.ParticleSwarm()

            optimal_params, _ = swarm_optimizer.optimize(ground_truth=raw_data,
                                                             contamination=self.infected_matrix,
                                                             selected_metrics=metrics, algorithm=self.algorithm,
                                                             n_particles=n_particles, c1=c1, c2=c2, w=w, iterations=iterations,n_processes=n_processes)

        elif optimizer == "sh":

            num_configs_d, num_iterations_d, reduction_factor_d, selected_metrics_d = defaults
            options = parameters.get('options', {})

            num_configs = options.get('num_configs', num_configs_d)
            num_iterations = options.get('num_iterations', num_iterations_d)
            reduction_factor = options.get('reduction_factor', reduction_factor_d)
            metrics = options.get('selected_metrics', selected_metrics_d)

            sh_optimizer = Optimization.SuccessiveHalving()

            optimal_params, _ = sh_optimizer.optimize(ground_truth=raw_data,
                                                             contamination=self.infected_matrix,
                                                             selected_metrics=metrics, algorithm=self.algorithm,
                                                             num_configs=num_configs, num_iterations=num_iterations, reduction_factor=reduction_factor)

        else:
            n_calls_d, selected_metrics_d = defaults
            options = parameters.get('options', {})

            n_calls = options.get('n_calls', n_calls_d)
            metrics = options.get('selected_metrics', selected_metrics_d)

            optimal_params, _ = Optimization.Greedy.optimize(ground_truth=raw_data,
                                                             contamination=self.infected_matrix,
                                                             selected_metrics=metrics, algorithm=self.algorithm,
                                                             n_calls=n_calls)

        self.parameters = optimal_params


class Imputation:

    def evaluate_params(ground_truth, contamination, configuration, algorithm="cdrec"):
        """
        evaluate various statistics for given parameters.
        :author: Quentin Nater

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
            algo.logs = False
            algo.impute(user_defined=True, params={"rank": rank, "epsilon": epsilon, "iterations": iterations})

        elif algorithm == 'iim':
            learning_neighbours = configuration[0]
            alg_code = "iim " + re.sub(r'[\W_]', '', str(learning_neighbours))

            algo = Imputation.Regression.IIM(contamination)
            algo.logs = False
            algo.impute(user_defined=True, params={"learning_neighbours": learning_neighbours, "alg_code": alg_code})

        elif algorithm == 'mrnn':
            hidden_dim, learning_rate, iterations = configuration

            algo = Imputation.ML.MRNN(contamination)
            algo.logs = False
            algo.impute(user_defined=True, params={"hidden_dim": hidden_dim, "learning_rate": learning_rate, "iterations": iterations, "seq_length": 7})

        elif algorithm == 'stmvl':
            window_size, gamma, alpha = configuration

            algo = Imputation.Pattern.STMVL(contamination)
            algo.logs = False
            algo.impute(user_defined=True, params={"window_size":window_size, "gamma": gamma, "alpha": alpha})

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
                :author: Quentin Nater

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
                :author: Quentin Nater

                :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

                :return: imputed_matrix, metrics : all time series with imputation data and their metrics
                """
                self.imputed_matrix = min_impute(self.infected_matrix, params)

                return self

    class MD:
        class CDRec(BaseImputer):
            """
            CDRec class to impute missing value with Centroid Decomposition.

            Methods
            -------
            impute(self, user_defined=True, params=None):
                Launch the imputation of the CDREC
            """

            algorithm = "cdrec"

            def impute(self, user_defined=True, params=None):
                """
                Imputation of data with CDREC algorithm.

                Parameters
                ----------
                user_defined : bool, optional
                    Decide whether to launch the imputation with default/user_defined parameters or auto-ml. Default is True.
                params : dict, optional
                    Parameters of the algorithm or Auto-ML, if None, default ones are loaded.

                    **Option 1: Algorithm parameters**

                    - rank : int
                        Rank of reduction of the matrix (must be higher than 1 and smaller than the limit of series).
                    - epsilon : float
                        Learning rate.
                    - iterations : int
                        Number of iterations.

                    **Option 2: AutoML parameters**

                    - ground_truth : numpy.ndarray
                        Values of time series without contamination.
                    - optimizer : str
                        Optimizer to use, can be "bayesian", "greedy", "pso", or "sh".
                    - options : dict, optional
                        Optional parameters for each optimizer.

                        **Bayesian:**

                        - n_calls : int, optional
                            Number of calls to the objective function. Default is 3.
                        - selected_metrics : list, optional
                            List of selected metrics to consider for optimization. Default is ["RMSE"].
                        - n_random_starts : int, optional
                            Number of initial calls to the objective function from random points. Default is 50.
                        - acq_func : str, optional
                            Function to minimize over the Gaussian prior, one of 'LCB', 'EI', 'PI', or 'gp_hedge'. Default is 'gp_hedge'.

                        **Greedy:**

                        - n_calls : int, optional
                            Number of calls to the objective function. Default is 3.
                        - selected_metrics : list, optional
                            List of selected metrics to consider for optimization. Default is ["RMSE"].

                        **PSO:**

                        - n_particles : int, optional
                            Number of particles used.
                        - c1 : float, optional
                            PSO c1 parameter.
                        - c2 : float, optional
                            PSO c2 parameter.
                        - w : float, optional
                            PSO w parameter.
                        - iterations : int, optional
                            Number of iterations for the optimization.
                        - n_processes : int, optional
                            Number of processes during the optimization.

                        **SH:**

                        - num_configs : int, optional
                            Number of configurations to try.
                        - num_iterations : int, optional
                            Number of iterations to run the optimization.
                        - reduction_factor : int, optional
                            Reduction factor for the number of configurations kept after each iteration.

                Returns
                -------
                self : CDRec
                    CDRec object with `imputed_matrix` set.

                :author: Quentin Nater
                """

                if params is not None:
                    rank, epsilon, iterations = self._check_params(user_defined, params)
                else:
                    rank, epsilon, iterations = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.imputed_matrix = cdrec(contamination=self.infected_matrix, truncation_rank=rank, iterations=iterations, epsilon=epsilon, logs=self.logs)

                return self

    class Regression:

        class IIM(BaseImputer):
            algorithm = "iim"

            def impute(self, user_defined=True, params=None):
                """
               Imputation of data with IIM algorithm
               :author: Quentin Nater

               :param params: [Optional] (neighbors, algo_code) : parameters of the algorithm, if None, default ones are loaded : neighbors, algo_code

               :return: imputed_matrix, metrics : all time series with imputation data and their metrics
               """
                if params is not None:
                    learning_neighbours, algo_code = self._check_params(user_defined, params)
                else:
                    learning_neighbours, algo_code = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.imputed_matrix = iim(contamination=self.infected_matrix, number_neighbor=learning_neighbours, algo_code=algo_code, logs=self.logs)

                return self

    class ML:
        class MRNN(BaseImputer):
            algorithm = "mrnn"

            def impute(self, user_defined=True, params=None):
                """
               Imputation of data with MRNN algorithm
               :author: Quentin Nater

               :param params: [Optional] (hidden_dim, learning_rate, iterations, sequence_length) : parameters of the algorithm, hidden_dim, learning_rate, iterations, keep_prob, sequence_length, if None, default ones are loaded

               :return: imputed_matrix, metrics : all time series with imputation data and their metrics
               """
                if params is not None:
                    hidden_dim, learning_rate, iterations, sequence_length = self._check_params(user_defined, params)
                else:
                    hidden_dim, learning_rate, iterations, sequence_length = utils.load_parameters(query="default", algorithm="mrnn")

                self.imputed_matrix = mrnn(contamination=self.infected_matrix, hidden_dim=hidden_dim,
                                           learning_rate=learning_rate, iterations=iterations,
                                           sequence_length=sequence_length, logs=self.logs)

                return self

    class Pattern:

        class STMVL(BaseImputer):
            algorithm = "stmvl"

            def impute(self, user_defined=True, params=None):
                """
               Imputation of data with MRNN algorithm
               :author: Quentin Nater

               :param params: [Optional] (window_size, gamma, alpha) : parameters of the algorithm, window_size, gamma, alpha, if None, default ones are loaded
                    :param window_size: window size for temporal component
                    :param gamma: smoothing parameter for temporal weight
                    :param alpha: power for spatial weight
               :return: imputed_matrix, metrics : all time series with imputation data and their metrics
               """
                if params is not None:
                    window_size, gamma, alpha = self._check_params(user_defined, params)
                else:
                    window_size, gamma, alpha = utils.load_parameters(query="default", algorithm="stmvl")

                self.imputed_matrix = stmvl(contamination=self.infected_matrix, window_size=window_size, gamma=gamma, alpha=alpha, logs=self.logs)

                return self
