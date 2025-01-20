import re

from imputegap.algorithms.brits import brits
from imputegap.algorithms.deep_mvi import deep_mvi
from imputegap.algorithms.dynammo import dynammo
from imputegap.algorithms.grouse import grouse
from imputegap.algorithms.iterative_svd import iterative_svd
from imputegap.algorithms.mean_impute import mean_impute
from imputegap.algorithms.mpin import mpin
from imputegap.algorithms.pristi import pristi
from imputegap.algorithms.rosl import rosl
from imputegap.algorithms.soft_impute import soft_impute
from imputegap.algorithms.spirit import spirit
from imputegap.algorithms.svt import svt
from imputegap.algorithms.tkcm import tkcm
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

        class IterativeSVD(BaseImputer):
            """
            IterativeSVD class to impute missing values using Iterative SVD.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the Iterative SDV algorithm.
            """

            algorithm = "iterative_svd"

            def impute(self, params=None):
                """
                Perform imputation using the Iterative SVD algorithm.

                Parameters
                ----------
                params : dict, optional
                    Parameters of the Iterative SVD algorithm or Auto-ML configuration, if None, default ones are loaded.

                    **Algorithm parameters:**

                    - rank : int
                        Rank of matrix reduction, which should be higher than 1 and smaller than the number of series.


                Returns
                -------
                self : IterativeSVD
                    IterativeSVD object with `recov_data` set.

                Example
                -------
                >>> i_svd_imputer = Imputation.MatrixCompletion.CDRec(incomp_data)
                >>> i_svd_imputer.impute()  # default parameters for imputation > or
                >>> i_svd_imputer.impute(params={'rank': 5})
                >>> recov_data = i_svd_imputer.recov_data

                References
                ----------
                Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor Hastie, Robert Tibshirani, David Botstein, Russ B. Altman, Missing value estimation methods for DNA microarrays , Bioinformatics, Volume 17, Issue 6, June 2001, Pages 520–525, https://doi.org/10.1093/bioinformatics/17.6.520
                """

                if params is not None:
                    rank = self._check_params(True, params)[0]
                else:
                    rank = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = iterative_svd(incomp_data=self.incomp_data, truncation_rank=rank, logs=self.logs)

                return self

        class GROUSE(BaseImputer):
            """
            GROUSE class to impute missing values using GROUSE.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the GROUSE algorithm.
            """

            algorithm = "grouse"

            def impute(self, params=None):
                """
                Perform imputation using the GROUSE algorithm.

                Parameters
                ----------
                params : dict, optional
                    Parameters of the GROUSE algorithm or Auto-ML configuration, if None, default ones are loaded.

                    **Algorithm parameters:**

                    - max_rank : int
                        Max rank of matrix reduction, which should be higher than 1 and smaller than the number of series.


                Returns
                -------
                self : GROUSE
                    GROUSE object with `recov_data` set.

                Example
                -------
                >>> grouse_imputer = Imputation.MatrixCompletion.GROUSE(incomp_data)
                >>> grouse_imputer.impute()  # default parameters for imputation > or
                >>> grouse_imputer.impute(params={'max_rank': 5})
                >>> recov_data = grouse_imputer.recov_data

                References
                ----------
                D. Zhang and L. Balzano. Global convergence of a grassmannian gradient descent algorithm for subspace estimation. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, AISTATS 2016, Cadiz, Spain, May 9-11, 2016, pages 1460–1468, 2016.
                """

                if params is not None:
                    max_rank = self._check_params(True, params)[0]
                else:
                    max_rank = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = grouse(incomp_data=self.incomp_data, max_rank=max_rank, logs=self.logs)

                return self

        class ROSL(BaseImputer):
            """
            ROSL class to impute missing values using Robust Online Subspace Learning algorithm.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the ROSL algorithm.
            """

            algorithm = "rosl"

            def impute(self, params=None):
                """
                Perform imputation using the ROSL algorithm.

                Parameters
                ----------
                params : dict, optional
                    Parameters of the ROSL algorithm or Auto-ML configuration, if None, default ones are loaded.

                    **Algorithm parameters:**

                     rank : int
                        The rank of the low-dimensional subspace for matrix decomposition.
                        Must be greater than 0 and less than or equal to the number of columns in the matrix.
                     regularization : float
                        The regularization parameter to control the trade-off between reconstruction accuracy and robustness.
                        Higher values enforce sparsity or robustness against noise in the data.

                Returns
                -------
                self : ROSL
                    ROSL object with `recov_data` set.

                Example
                -------
                >>> rosl_imputer = Imputation.MatrixCompletion.ROSL(incomp_data)
                >>> rosl_imputer.impute()  # default parameters for imputation > or
                >>> rosl_imputer.impute(params={'rank': 5, 'regularization': 10})
                >>> recov_data = rosl_imputer.recov_data

                References
                ----------
                X. Shu, F. Porikli, and N. Ahuja. Robust orthonormal subspace learning: Efficient recovery of corrupted low-rank matrices. In 2014 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2014, Columbus, OH, USA, June 23-28, 2014, pages 3874–3881, 2014.
                """
                if params is not None:
                    rank, regularization = self._check_params(True, params)
                else:
                    rank, regularization = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = rosl(incomp_data=self.incomp_data, rank=rank, regularization=regularization, logs=self.logs)

                return self

        class SoftImpute(BaseImputer):
            """
            SoftImpute class to impute missing values using Soft Impute algorithm.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the Soft Impute algorithm.
            """

            algorithm = "soft_impute"

            def impute(self, params=None):
                """
                Perform imputation using the Soft Impute algorithm.

                Parameters
                ----------
                params : dict, optional
                    Parameters of the Soft Impute algorithm or Auto-ML configuration, if None, default ones are loaded.

                    **Algorithm parameters:**

                     max_rank : int
                        The max rank of the low-dimensional subspace for matrix decomposition.
                        Must be greater than 0 and less than or equal to the number of columns in the matrix.

                Returns
                -------
                self : SoftImpute
                    SoftImpute object with `recov_data` set.

                Example
                -------
                >>> soft_impute_imputer = Imputation.MatrixCompletion.SoftImpute(incomp_data)
                >>> soft_impute_imputer.impute()  # default parameters for imputation > or
                >>> soft_impute_imputer.impute(params={'max_rank': 5})
                >>> recov_data = soft_impute_imputer.recov_data

                References
                ----------
                R. Mazumder, T. Hastie, and R. Tibshirani. Spectral regularization algorithms for learning large incomplete matrices. Journal of Machine Learning Research, 11:2287–2322, 2010.
                """
                if params is not None:
                    max_rank = self._check_params(True, params)[0]
                else:
                    max_rank = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = soft_impute(incomp_data=self.incomp_data, max_rank=max_rank, logs=self.logs)

                return self

        class SPIRIT(BaseImputer):
            """
            SPIRIT class to impute missing values using SPIRIT algorithm.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the SPIRIT algorithm.
            """

            algorithm = "spirit"

            def impute(self, params=None):
                """
                Perform imputation using the SPIRIT algorithm.

                Parameters
                ----------
                params : dict, optional
                    Parameters of the SPIRIT algorithm or Auto-ML configuration, if None, default ones are loaded.

                    **Algorithm parameters:**

                    k : int
                        The number of eigencomponents (principal components) to retain for dimensionality reduction.
                        Example: 2, 5, 10.
                    w : int
                        The window size for capturing temporal dependencies.
                        Example: 5 (short-term), 20 (long-term).
                    lambda_value : float
                        The forgetting factor controlling how quickly past data is "forgotten".
                        Example: 0.8 (fast adaptation), 0.95 (stable systems).

                Returns
                -------
                self : SPIRIT
                    SPIRIT object with `recov_data` set.

                Example
                -------
                >>> spirit_imputer = Imputation.MatrixCompletion.SPIRIT(incomp_data)
                >>> spirit_imputer.impute()  # default parameters for imputation > or
                >>> spirit_imputer.impute(params={'k': 2, 'w': 5, 'lambda_value': 0.85})
                >>> recov_data = spirit_imputer.recov_data

                References
                ----------
                S. Papadimitriou, J. Sun, and C. Faloutsos. Streaming pattern discovery in multiple time-series. In Proceedings of the 31st International Conference on Very Large Data Bases, Trondheim, Norway, August 30 - September 2, 2005, pages 697–708, 2005.
                """
                if params is not None:
                    k, w, lambda_value = self._check_params(True, params)
                else:
                    k, w, lambda_value = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = spirit(incomp_data=self.incomp_data, k=k, w=w, lambda_value=lambda_value, logs=self.logs)

                return self

        class SVT(BaseImputer):
            """
            SVT class to impute missing values using Singular Value Thresholding algorithm.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the SVT algorithm.
            """

            algorithm = "svt"

            def impute(self, params=None):
                """
                Perform imputation using the SVT algorithm.

                Parameters
                ----------
                params : dict, optional
                    Parameters of the SVT algorithm or Auto-ML configuration, if None, default ones are loaded.

                    **Algorithm parameters:**

                    tau : float
                        The thresholding parameter for singular values. Controls how singular values are shrunk during the decomposition process.
                        Larger values encourage a sparser, lower-rank solution, while smaller values retain more detail.


                Returns
                -------
                self : SVT
                    SVT object with `recov_data` set.

                Example
                -------
                >>> svt_imputer = Imputation.MatrixCompletion.SVT(incomp_data)
                >>> svt_imputer.impute()  # default parameters for imputation > or
                >>> svt_imputer.impute(params={'tau': 1})
                >>> recov_data = svt_imputer.recov_data

                References
                ----------
                J. Cai, E. J. Candès, and Z. Shen. A singular value thresholding algorithm for matrix completion. SIAM Journal on Optimization, 20(4):1956–1982, 2010. [8] J. Cambronero, J. K. Feser, M. J. Smith, and
                """
                if params is not None:
                    tau = self._check_params(True, params)[0]
                else:
                    tau = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = svt(incomp_data=self.incomp_data, tau=tau, logs=self.logs)

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

        class DynaMMo(BaseImputer):
            """
            DynaMMo class to impute missing values using Dynamic Multi-Mode modeling with Missing Observations algorithm.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the DynaMMo algorithm.
            """

            algorithm = "dynammo"

            def impute(self, params=None):
                """
                Perform imputation using the DynaMMo algorithm.

                Parameters
                ----------
                params : dict, optional
                    Parameters of the DynaMMo algorithm or Auto-ML configuration, if None, default ones are loaded.

                    **Algorithm parameters:**

                        h : int
                            The time window (H) parameter for modeling temporal dynamics.
                        max_iteration : int
                            The maximum number of iterations for the imputation process.
                        approximation : bool
                            If True, enables faster approximate processing.

                Returns
                -------
                self : DynaMMo
                    DynaMMo object with `recov_data` set.

                Example
                -------
                >>> dynammo_imputer = Imputation.PatternSearch.DynaMMo(incomp_data)
                >>> dynammo_imputer.impute()  # default parameters for imputation > or
                >>> dynammo_imputer.impute(params={'h': 5, 'max_iteration': 100, 'approximation': True})
                >>> recov_data = dynammo_imputer.recov_data

                References
                ----------
                L. Li, J. McCann, N. S. Pollard, and C. Faloutsos. Dynammo: mining and summarization of coevolving sequences with missing values. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Paris, France, June 28 - July 1, 2009, pages 507–516, 2009.
                """
                if params is not None:
                    h, max_iteration, approximation = self._check_params(True, params)
                else:
                    h, max_iteration, approximation = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = dynammo(incomp_data=self.incomp_data, h=h, max_iteration=max_iteration,
                                          approximation=approximation, logs=self.logs)

                return self

        class TKCM(BaseImputer):
            """
            TKCM class to impute missing values using Tensor Kernelized Coupled Matrix Completion algorithm.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the TKCM algorithm.
            """

            algorithm = "tkcm"

            def impute(self, params=None):
                """
                Perform imputation using the TKCM algorithm.

                Parameters
                ----------
                params : dict, optional
                    Parameters of the TKCM algorithm or Auto-ML configuration, if None, default ones are loaded.

                    **Algorithm parameters:**

                    rank : int
                        The rank for matrix decomposition (must be greater than 1 and smaller than the number of series).

                Returns
                -------
                self : TKCM
                    TKCM object with `recov_data` set.

                Example
                -------
                >>> tkcm_imputer = Imputation.PatternSearch.TKCM(incomp_data)
                >>> tkcm_imputer.impute()  # default parameters for imputation > or
                >>> tkcm_imputer.impute(params={'rank': 5})
                >>> recov_data = tkcm_imputer.recov_data

                References
                ----------
                K. Wellenzohn, M. H. Böhlen, A. Dignös, J. Gamper, and H. Mitterer. Continuous imputation of missing values in streams of pattern-determining time series. In Proceedings of the 20th International Conference on Extending Database Technology, EDBT 2017, Venice, Italy, March 21-24, 2017., pages 330–341, 2017.
                """
                if params is not None:
                    rank = self._check_params(True, params)[0]
                else:
                    rank = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = tkcm(incomp_data=self.incomp_data, rank=rank, logs=self.logs)

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
                    hidden_dim, learning_rate, iterations, sequence_length = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = mrnn(incomp_data=self.incomp_data, hidden_dim=hidden_dim,
                                       learning_rate=learning_rate, iterations=iterations,
                                       sequence_length=sequence_length, logs=self.logs)

                return self

        class BRITS(BaseImputer):
            """
            BRITS class to impute missing values using Bidirectional Recurrent Imputation for Time Series

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the BRITS algorithm.
            """
            algorithm = "brits"

            def impute(self, user_def=True, params=None):
                """
                Perform imputation using the BRITS algorithm.

                Parameters
                ----------
                user_def : bool, optional
                    Whether to use user-defined or default parameters (default is True).
                params : dict, optional
                    Parameters of the BRITS algorithm, if None, default ones are loaded.

                    - model : str
                        Specifies the type of model to use for the imputation. Options may include predefined models like 'brits', 'brits-i' or 'brits_i_univ'.
                    - epoch : int
                        Number of epochs for training the model. Determines how many times the algorithm processes the entire dataset during training.
                    - batch_size : int
                        Size of the batches used during training. Larger batch sizes can speed up training but may require more memory.
                    - nbr_features : int
                        Number of features, dimension in the time series.
                    - hidden_layer : int
                        Number of units in the hidden layer of the model. Controls the capacity of the neural network to learn complex patterns.

                Returns
                -------
                self : BRITS
                    The object with `recov_data` set.

                Example
                -------
                >>> brits_imputer = Imputation.DeepLearning.BRITS(incomp_data)
                >>> brits_imputer.impute()  # default parameters for imputation
                >>> recov_data = brits_imputer.recov_data

                References
                ----------
                Cao, W., Wang, D., Li, J., Zhou, H., Li, L. & Li, Y. BRITS: Bidirectional Recurrent Imputation for Time Series. Advances in Neural Information Processing Systems, 31 (2018). https://proceedings.neurips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf
                """
                if params is not None:
                    model, epoch, batch_size, nbr_features, hidden_layer = self._check_params(True, params)
                else:
                    model, epoch, batch_size, nbr_features, hidden_layer = utils.load_parameters(query="default", algorithm=self.algorithm)

                seq_length = self.incomp_data.shape[1]

                self.recov_data = brits(incomp_data=self.incomp_data, model=model, epoch=epoch, batch_size=batch_size, nbr_features=nbr_features, hidden_layers=hidden_layer, seq_length=seq_length, logs=self.logs)
                return self

        class DeepMVI(BaseImputer):
            """
            DeepMVI class to impute missing values using Deep Multivariate Imputation

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the DeepMVI algorithm.
            """
            algorithm = "deep_mvi"

            def impute(self, user_def=True, params=None):
                """
                Perform imputation using the DeepMVI algorithm.

                Parameters
                ----------


                Returns
                -------
                self : DeepMVI
                    The object with `recov_data` set.

                Example
                -------
                >>> deep_mvi_imputer = Imputation.DeepLearning.DeepMVI(incomp_data)
                >>> deep_mvi_imputer.impute()  # default parameters for imputation
                >>> recov_data = deep_mvi_imputer.recov_data

                References
                ----------
                P. Bansal, P. Deshpande, and S. Sarawagi. Missing value imputation on multidimensional time series. arXiv preprint arXiv:2103.01600, 2023
                """
                if params is not None:
                    max_epoch, patience = self._check_params(True, params)
                else:
                    max_epoch, patience = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = deep_mvi(incomp_data=self.incomp_data, max_epoch=max_epoch, patience=patience, logs=self.logs)
                return self

        class MPIN(BaseImputer):
            """
            MPIN class to impute missing values using Multi-attribute Sensor Data Streams via Message Propagation algorithm.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the MPIN algorithm.
            """
            algorithm = "mpin"

            def impute(self, user_def=True, params=None):
                """
                Perform imputation using the MPIN algorithm.

                Parameters
                ----------


                Returns
                -------
                self : MPIN
                    The object with `recov_data` set.

                Example
                -------
                >>> deep_mvi_imputer = Imputation.DeepLearning.DeepMVI(incomp_data)
                >>> deep_mvi_imputer.impute()  # default parameters for imputation
                >>> recov_data = deep_mvi_imputer.recov_data

                References
                ----------
                Li, X., Li, H., Lu, H., Jensen, C.S., Pandey, V. & Markl, V. Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation (Extended Version). arXiv (2023). https://arxiv.org/abs/2311.07344
                """
                if params is not None:
                    incre_mode, window, k, learning_rate, weight_decay, epochs, threshold, base = self._check_params(True, params)
                else:
                    incre_mode, window, k, learning_rate, weight_decay, epochs, threshold, base = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = mpin(incomp_data=self.incomp_data, incre_mode=incre_mode, window=window, k=k, lr=learning_rate, weight_decay=weight_decay, epochs=epochs, thre=threshold, base=base, logs=self.logs)
                return self

        class PRISTI(BaseImputer):
            """
            PRISTI class to impute missing values using A Conditional Diffusion Framework for Spatiotemporal Imputation algorithm.

            Methods
            -------
            impute(self, user_def=True, params=None):
                Perform imputation using the PRISTI algorithm.
            """
            algorithm = "pristi"

            def impute(self, user_def=True, params=None):
                """
                Perform imputation using the PRISTI algorithm.

                Parameters
                ----------


                Returns
                -------
                self : PRISTI
                    The object with `recov_data` set.

                Example
                -------
                >>> pristi_imputer = Imputation.DeepLearning.PRISTI(incomp_data)
                >>> pristi_imputer.impute()  # default parameters for imputation
                >>> recov_data = pristi_imputer.recov_data

                References
                ----------
                M. Liu, H. Huang, H. Feng, L. Sun, B. Du and Y. Fu, "PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation," 2023 IEEE 39th International Conference on Data Engineering (ICDE), Anaheim, CA, USA, 2023, pp. 1927-1939, doi: 10.1109/ICDE55515.2023.00150.
                """
                if params is not None:
                    target_strategy, unconditional, seed, device = self._check_params(True, params)
                else:
                    target_strategy, unconditional, seed, device = utils.load_parameters(query="default", algorithm=self.algorithm)

                self.recov_data = pristi(incomp_data=self.incomp_data, target_strategy=target_strategy, unconditional=unconditional, seed=seed, device=device, logs=self.logs)
                return self


    class GraphLearning:
        """
        A class containing imputation algorithms for graph-learning-based methods.
        TO COME SOON...

        Subclasses
        ----------
        """

