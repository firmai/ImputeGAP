import re

from imputegap.algorithms.mean_impute import mean_impute
from imputegap.recovery.evaluation import Evaluation
from imputegap.algorithms.cdrec import cdrec
from imputegap.algorithms.iim import iim
from imputegap.algorithms.min_impute import min_impute
from imputegap.algorithms.mrnn import mrnn
from imputegap.algorithms.stmvl import stmvl
from imputegap.algorithms.zero_impute import zero_impute
from imputegap.tools import utils


class BaseImputer:
    """
    Base class for imputation algorithms.

    This class provides common methods for imputation tasks such as scoring, parameter checking,
    and optimization. Specific algorithms should inherit from this class and implement the `impute` method.

    Methods
    -------
    impute(params=None):
        Abstract method to perform the imputation.
    score(input_data, recov_data=None):
        Compute metrics for the imputed time series.
    _check_params(user_def, params):
        Check and format parameters for imputation.
    _optimize(parameters={}):
        Optimize hyperparameters for the imputation algorithm.
    """
    algorithm = ""  # Class variable to hold the algorithm name
    logs = True

    def __init__(self, incomp_data):
        """
        Initialize the BaseImputer with an infected time series matrix.

        Parameters
        ----------
        incomp_data : numpy.ndarray
            Matrix used during the imputation of the time series.
        """
        self.incomp_data = incomp_data
        self.recov_data = None
        self.metrics = None
        self.parameters = None

    def impute(self, params=None):
        """
        Abstract method to perform the imputation. Must be implemented in subclasses.

        Parameters
        ----------
        params : dict, optional
            Dictionary of algorithm parameters (default is None).

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def score(self, input_data, recov_data=None):
        """
        Compute evaluation metrics for the imputed time series.

        Parameters
        ----------
        input_data : numpy.ndarray
            The original time series without contamination.
        recov_data : numpy.ndarray, optional
            The imputed time series (default is None).

        Returns
        -------
        None
        """
        if self.recov_data is None:
            self.recov_data = recov_data

        self.metrics = Evaluation(input_data, self.recov_data, self.incomp_data).compute_all_metrics()

    def _check_params(self, user_def, params):
        """
        Format the parameters for optimization or imputation.

        Parameters
        ----------
        user_def : bool
            Whether the parameters are user-defined or not.
        params : dict or list
            List or dictionary of parameters.

        Returns
        -------
        tuple
            Formatted parameters as a tuple.
        """

        if params is not None:
            if not user_def:
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
        Conduct the optimization of the hyperparameters using different optimizers.

        Parameters
        ----------
        parameters : dict
            Dictionary containing optimization configurations such as input_data, optimizer, and options.

        Returns
        -------
        None
        """
        from imputegap.recovery.optimization import Optimization

        input_data = parameters.get('input_data')
        if input_data is None:
            raise ValueError(f"Need input_data to be able to adapt the hyper-parameters: {input_data}")

        optimizer = parameters.get('optimizer', "bayesian")
        defaults = utils.load_parameters(query="default", algorithm=optimizer)

        print("\noptimizer", optimizer, "has been called with", self.algorithm, "...\n")

        if optimizer == "bayesian":
            n_calls_d, n_random_starts_d, acq_func_d, selected_metrics_d = defaults
            options = parameters.get('options', {})

            n_calls = options.get('n_calls', n_calls_d)
            random_starts = options.get('n_random_starts', n_random_starts_d)
            func = options.get('acq_func', acq_func_d)
            metrics = options.get('metrics', selected_metrics_d)

            bo_optimizer = Optimization.Bayesian()

            optimal_params, _ = bo_optimizer.optimize(input_data=input_data,
                                                      incomp_data=self.incomp_data,
                                                      metrics=metrics,
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
            metrics = options.get('metrics', selected_metrics_d)

            swarm_optimizer = Optimization.ParticleSwarm()

            optimal_params, _ = swarm_optimizer.optimize(input_data=input_data,
                                                         incomp_data=self.incomp_data,
                                                         metrics=metrics, algorithm=self.algorithm,
                                                         n_particles=n_particles, c1=c1, c2=c2, w=w,
                                                         iterations=iterations, n_processes=n_processes)

        elif optimizer == "sh":

            num_configs_d, num_iterations_d, reduction_factor_d, selected_metrics_d = defaults
            options = parameters.get('options', {})

            num_configs = options.get('num_configs', num_configs_d)
            num_iterations = options.get('num_iterations', num_iterations_d)
            reduction_factor = options.get('reduction_factor', reduction_factor_d)
            metrics = options.get('metrics', selected_metrics_d)

            sh_optimizer = Optimization.SuccessiveHalving()

            optimal_params, _ = sh_optimizer.optimize(input_data=input_data,
                                                      incomp_data=self.incomp_data,
                                                      metrics=metrics, algorithm=self.algorithm,
                                                      num_configs=num_configs, num_iterations=num_iterations,
                                                      reduction_factor=reduction_factor)

        else:
            n_calls_d, selected_metrics_d = defaults
            options = parameters.get('options', {})

            n_calls = options.get('n_calls', n_calls_d)
            metrics = options.get('metrics', selected_metrics_d)

            go_optimizer = Optimization.Greedy()

            optimal_params, _ = go_optimizer.optimize(input_data=input_data,
                                                      incomp_data=self.incomp_data,
                                                      metrics=metrics, algorithm=self.algorithm,
                                                      n_calls=n_calls)

        self.parameters = optimal_params


class Imputation:
    """
    A class containing static methods for evaluating and running imputation algorithms on time series data.

    Methods
    -------
    evaluate_params(input_data, incomp_data, configuration, algorithm="cdrec"):
        Evaluate imputation performance using given parameters and algorithm.
    """

    def evaluate_params(input_data, incomp_data, configuration, algorithm="cdrec"):
        """
        Evaluate various metrics for given parameters and imputation algorithm.

        Parameters
        ----------
        input_data : numpy.ndarray
            The original time series without contamination.
        incomp_data : numpy.ndarray
            The time series with contamination.
        configuration : tuple
            Tuple of the configuration of the algorithm.
        algorithm : str, optional
            Imputation algorithm to use. Valid values: 'cdrec', 'mrnn', 'stmvl', 'iim' (default is 'cdrec').

        Returns
        -------
        dict
            A dictionary of computed evaluation metrics.
        """

        if isinstance(configuration, dict):
            configuration = tuple(configuration.values())

        if algorithm == 'cdrec':
            rank, epsilon, iterations = configuration
            algo = Imputation.MatrixCompletion.CDRec(incomp_data)
            algo.logs = False
            algo.impute(user_def=True, params={"rank": rank, "epsilon": epsilon, "iterations": iterations})

        elif algorithm == 'iim':
            if not isinstance(configuration, list):
                configuration = [configuration]
            learning_neighbours = configuration[0]
            alg_code = "iim " + re.sub(r'[\W_]', '', str(learning_neighbours))

            algo = Imputation.Statistics.IIM(incomp_data)
            algo.logs = False
            algo.impute(user_def=True, params={"learning_neighbours": learning_neighbours, "alg_code": alg_code})

        elif algorithm == 'mrnn':
            hidden_dim, learning_rate, iterations = configuration

            algo = Imputation.DeepLearning.MRNN(incomp_data)
            algo.logs = False
            algo.impute(user_def=True,
                        params={"hidden_dim": hidden_dim, "learning_rate": learning_rate, "iterations": iterations,
                                "seq_length": 7})

        elif algorithm == 'stmvl':
            window_size, gamma, alpha = configuration

            algo = Imputation.PatternSearch.STMVL(incomp_data)
            algo.logs = False
            algo.impute(user_def=True, params={"window_size": window_size, "gamma": gamma, "alpha": alpha})

        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

        algo.score(input_data)
        error_measures = algo.metrics

        return error_measures

    class Statistics:
        """
        A class containing specific imputation algorithms for statistical methods.

        Subclasses
        ----------
        ZeroImpute :
            Imputation method that replaces missing values with zeros.
        MinImpute :
            Imputation method that replaces missing values with the minimum value of the ground truth.
        """

        class ZeroImpute(BaseImputer):
            """
            ZeroImpute class to impute missing values with zeros.

            Methods
            -------
            impute(self, params=None):
                Perform imputation by replacing missing values with zeros.
            """
            algorithm = "zero_impute"

            def impute(self, params=None):
                """
                Impute missing values by replacing them with zeros.
                Template for adding external new algorithm

                Parameters
                ----------
                params : dict, optional
                    Dictionary of algorithm parameters (default is None).

                Returns
                -------
                self : ZeroImpute
                    The object with `recov_data` set.
                """
                self.recov_data = zero_impute(self.incomp_data, params)

                return self

        class MinImpute(BaseImputer):
            """
            MinImpute class to impute missing values with the minimum value of the ground truth.

            Methods
            -------
            impute(self, params=None):
                Perform imputation by replacing missing values with the minimum value of the ground truth.
            """
            algorithm = "min_impute"

            def impute(self, params=None):
                """
                Impute missing values by replacing them with the minimum value of the ground truth.
                Template for adding external new algorithm

                Parameters
                ----------
                params : dict, optional
                    Dictionary of algorithm parameters (default is None).

                Returns
                -------
                self : MinImpute
                    The object with `recov_data` set.
                """
                self.recov_data = min_impute(self.incomp_data, params)

                return self

        class MeanImpute(BaseImputer):
            """
            MeanImpute class to impute missing values with the mean value of the ground truth.

            Methods
            -------
            impute(self, params=None):
                Perform imputation by replacing missing values with the mean value of the ground truth.
            """
            algorithm = "mean_impute"

            def impute(self, params=None):
                """
                Impute missing values by replacing them with the mean value of the ground truth.
                Template for adding external new algorithm

                Parameters
                ----------
                params : dict, optional
                    Dictionary of algorithm parameters (default is None).

                Returns
                -------
                self : MinImpute
                    The object with `recov_data` set.
                """
                self.recov_data = mean_impute(self.incomp_data, params)

                return self

        class IIM(BaseImputer):
            """
            IIM class to impute missing values using Iterative Imputation with Metric Learning (IIM).

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the IIM algorithm.
            """
            algorithm = "iim"

            def impute(self, user_def=True, params=None):
                """
                Perform imputation using the IIM algorithm.

                Parameters
                ----------
                user_def : bool, optional
                    Whether to use user-defined or default parameters (default is True).
                params : dict, optional
                    Parameters of the IIM algorithm, if None, default ones are loaded.

                    - learning_neighbours : int
                        Number of nearest neighbors for learning.
                    - algo_code : str
                        Unique code for the algorithm configuration.

                Returns
                -------
                self : IIM
                    The object with `recov_data` set.

                Example
                -------
                >>> iim_imputer = Imputation.Statistics.IIM(incomp_data)
                >>> iim_imputer.impute()  # default parameters for imputation > or
                >>> iim_imputer.impute(user_def=True, params={'learning_neighbors': 10})  # user-defined  > or
                >>> iim_imputer.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 2}})  # auto-ml with bayesian
                >>> recov_data = iim_imputer.recov_data

                References
                ----------
                A. Zhang, S. Song, Y. Sun and J. Wang, "Learning Individual Models for Imputation," 2019 IEEE 35th International Conference on Data Engineering (ICDE), Macao, China, 2019, pp. 160-171, doi: 10.1109/ICDE.2019.00023.
                keywords: {Data models;Adaptation models;Computational modeling;Predictive models;Numerical models;Aggregates;Regression tree analysis;Missing values;Data imputation}
                """
                if params is not None:
                    learning_neighbours, algo_code = self._check_params(user_def, params)
                else:
                    learning_neighbours, algo_code = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = iim(incomp_data=self.incomp_data, number_neighbor=learning_neighbours,
                                      algo_code=algo_code, logs=self.logs)

                return self

    class MatrixCompletion:
        """
        A class containing imputation algorithms for matrix decomposition methods.

        Subclasses
        ----------
        CDRec :
            Imputation method using Centroid Decomposition.
        """

        class CDRec(BaseImputer):
            """
            CDRec class to impute missing values using Centroid Decomposition (CDRec).

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the CDRec algorithm.
            """

            algorithm = "cdrec"

            def impute(self, user_def=True, params=None):
                """
                Perform imputation using the CDRec algorithm.

                Parameters
                ----------
                user_def : bool, optional
                    Whether to use user-defined or default parameters (default is True).
                params : dict, optional
                    Parameters of the CDRec algorithm or Auto-ML configuration, if None, default ones are loaded.

                    **Algorithm parameters:**

                    - rank : int
                        Rank of matrix reduction, which should be higher than 1 and smaller than the number of series.
                    - epsilon : float
                        The learning rate used for the algorithm.
                    - iterations : int
                        The number of iterations to perform.

                    **Auto-ML parameters:**

                    - input_data : numpy.ndarray
                        The original time series dataset without contamination.
                    - optimizer : str
                        The optimizer to use for parameter optimization. Valid values are "bayesian", "greedy", "pso", or "sh".
                    - options : dict, optional
                        Optional parameters specific to the optimizer.

                        **Bayesian:**

                        - n_calls : int, optional
                            Number of calls to the objective function. Default is 3.
                        - metrics : list, optional
                            List of selected metrics to consider for optimization. Default is ["RMSE"].
                        - n_random_starts : int, optional
                            Number of initial calls to the objective function, from random points. Default is 50.
                        - acq_func : str, optional
                            Acquisition function to minimize over the Gaussian prior. Valid values: 'LCB', 'EI', 'PI', 'gp_hedge' (default is 'gp_hedge').

                        **Greedy:**

                        - n_calls : int, optional
                            Number of calls to the objective function. Default is 3.
                        - metrics : list, optional
                            List of selected metrics to consider for optimization. Default is ["RMSE"].

                        **PSO:**

                        - n_particles : int, optional
                            Number of particles used.
                        - c1 : float, optional
                            PSO learning coefficient c1 (personal learning).
                        - c2 : float, optional
                            PSO learning coefficient c2 (global learning).
                        - w : float, optional
                            PSO inertia weight.
                        - iterations : int, optional
                            Number of iterations for the optimization.
                        - n_processes : int, optional
                            Number of processes during optimization.

                        **Successive Halving (SH):**

                        - num_configs : int, optional
                            Number of configurations to try.
                        - num_iterations : int, optional
                            Number of iterations to run the optimization.
                        - reduction_factor : int, optional
                            Reduction factor for the number of configurations kept after each iteration.

                Returns
                -------
                self : CDRec
                    CDRec object with `recov_data` set.

                Example
                -------
                >>> cdrec_imputer = Imputation.MatrixCompletion.CDRec(incomp_data)
                >>> cdrec_imputer.impute()  # default parameters for imputation > or
                >>> cdrec_imputer.impute(user_def=True, params={'rank': 5, 'epsilon': 0.01, 'iterations': 100})  # user-defined > or
                >>> cdrec_imputer.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 2}})  # auto-ml with bayesian
                >>> recov_data = cdrec_imputer.recov_data

                References
                ----------
                Khayati, M., Cudré-Mauroux, P. & Böhlen, M.H. Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl Inf Syst 62, 2257–2280 (2020). https://doi.org/10.1007/s10115-019-01421-7
                """

                if params is not None:
                    rank, epsilon, iterations = self._check_params(user_def, params)
                else:
                    rank, epsilon, iterations = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = cdrec(incomp_data=self.incomp_data, truncation_rank=rank,
                                        iterations=iterations, epsilon=epsilon, logs=self.logs)

                return self


    class DeepLearning:
        """
        A class containing imputation algorithms for deep learning-based methods.

        Subclasses
        ----------
        MRNN :
            Imputation method using Multi-directional Recurrent Neural Networks (MRNN).
        """

        class MRNN(BaseImputer):
            """
            MRNN class to impute missing values using Multi-directional Recurrent Neural Networks (MRNN).

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the MRNN algorithm.
            """
            algorithm = "mrnn"

            def impute(self, user_def=True, params=None):
                """
                Perform imputation using the MRNN algorithm.

                Parameters
                ----------
                user_def : bool, optional
                    Whether to use user-defined or default parameters (default is True).
                params : dict, optional
                    Parameters of the MRNN algorithm, if None, default ones are loaded.

                    - hidden_dim : int
                        The number of hidden units in the neural network.
                    - learning_rate : float
                        Learning rate for training the neural network.
                    - iterations : int
                        Number of iterations for training.
                    - sequence_length : int
                        The length of the sequences used in the recurrent neural network.

                Returns
                -------
                self : MRNN
                    The object with `recov_data` set.

                Example
                -------
                >>> mrnn_imputer = Imputation.DeepLearning.MRNN(incomp_data)
                >>> mrnn_imputer.impute()  # default parameters for imputation > or
                >>> mrnn_imputer.impute(user_def=True, params={'hidden_dim': 10, 'learning_rate':0.01, 'iterations':50, 'sequence_length': 7})  # user-defined > or
                >>> mrnn_imputer.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 2}})  # auto-ml with bayesian
                >>> recov_data = mrnn_imputer.recov_data

                References
                ----------
                J. Yoon, W. R. Zame and M. van der Schaar, "Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks," in IEEE Transactions on Biomedical Engineering, vol. 66, no. 5, pp. 1477-1490, May 2019, doi: 10.1109/TBME.2018.2874712. keywords: {Time measurement;Interpolation;Estimation;Medical diagnostic imaging;Correlation;Recurrent neural networks;Biomedical measurement;Missing data;temporal data streams;imputation;recurrent neural nets}
                """
                if params is not None:
                    hidden_dim, learning_rate, iterations, sequence_length = self._check_params(user_def, params)
                else:
                    hidden_dim, learning_rate, iterations, sequence_length = utils.load_parameters(query="default",
                                                                                                   algorithm="mrnn")

                self.recov_data = mrnn(incomp_data=self.incomp_data, hidden_dim=hidden_dim,
                                       learning_rate=learning_rate, iterations=iterations,
                                       sequence_length=sequence_length, logs=self.logs)

                return self

    class PatternSearch:
        """
        A class containing imputation algorithms for pattern-based methods.

        Subclasses
        ----------
        STMVL :
            Imputation method using Spatio-Temporal Matrix Variational Learning (STMVL).
        """

        class STMVL(BaseImputer):
            """
            STMVL class to impute missing values using Spatio-Temporal Matrix Variational Learning (STMVL).

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the STMVL algorithm.
            """
            algorithm = "stmvl"

            def impute(self, user_def=True, params=None):
                """
                Perform imputation using the STMVL algorithm.

                Parameters
                ----------
                user_def : bool, optional
                    Whether to use user-defined or default parameters (default is True).
                params : dict, optional
                    Parameters of the STMVL algorithm, if None, default ones are loaded.

                    - window_size : int
                        The size of the temporal window for imputation.
                    - gamma : float
                        Smoothing parameter for temporal weights.
                    - alpha : float
                        Power for spatial weights.

                Returns
                -------
                self : STMVL
                    The object with `recov_data` set.

                Example
                -------
                >>> stmvl_imputer = Imputation.PatternSearch.STMVL(incomp_data)
                >>> stmvl_imputer.impute()  # default parameters for imputation > or
                >>> stmvl_imputer.impute(user_def=True, params={'window_size': 7, 'learning_rate':0.01, 'gamma':0.85, 'alpha': 7})  # user-defined  > or
                >>> stmvl_imputer.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 2}})  # auto-ml with bayesian
                >>> recov_data = stmvl_imputer.recov_data

                References
                ----------
                Yi, X., Zheng, Y., Zhang, J., & Li, T. ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data.
                School of Information Science and Technology, Southwest Jiaotong University; Microsoft Research; Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences.
                """
                if params is not None:
                    window_size, gamma, alpha = self._check_params(user_def, params)
                else:
                    window_size, gamma, alpha = utils.load_parameters(query="default", algorithm="stmvl")

                self.recov_data = stmvl(incomp_data=self.incomp_data, window_size=window_size, gamma=gamma,
                                        alpha=alpha, logs=self.logs)

                return self

    class GraphLearning:
        """
        A class containing imputation algorithms for graph-learning-based methods.
        TO COME SOON...

        Subclasses
        ----------
        """

