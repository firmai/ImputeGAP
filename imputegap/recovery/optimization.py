import os
import toml
from itertools import product
import numpy as np


import skopt
from skopt.utils import use_named_args
from skopt.space import Integer

from imputegap.recovery.imputation import Imputation
from imputegap.tools.algorithm_parameters import SEARCH_SPACES, ALL_ALGO_PARAMS

# Define the search space for each algorithm separately
search_spaces = SEARCH_SPACES


class Optimization:

    def save_optimization(optimal_params, algorithm="cdrec", dataset="", optimizer="b", file_name=None):
        """
        Save the optimization parameters to a TOML file to use later without recomputing.

        :param optimal_params: dictionary of the optimal parameters.
        :param file_name: name of the TOML file to save the results. Default is 'optimization_results.toml'.
        """
        if file_name is None:
            file_name = "../params/optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"

        if not os.path.exists(file_name):
            file_name = file_name[1:]

        params_to_save = {algorithm: optimal_params}

        try:
            with open(file_name, 'w') as file:
                toml.dump(params_to_save, file)
            print(f"\nOptimization parameters successfully saved to {file_name}")
        except Exception as e:
            print(f"\nAn error occurred while saving the file: {e}")

    class Greedy:

        def optimize(ground_truth, contamination, selected_metrics=["RMSE"], algorithm="cdrec", n_calls=250):
            """
            Conduct the Greedy optimization hyperparameter optimization.

            Parameters
            ----------
            :param ground_truth : time series data set to optimize
            :param contamination : time series contaminate to impute
            :param selected_metrics : list of selected metrics to consider for optimization. | default ["RMSE"]
            :param algorithm : imputation algorithm | Valid values: 'cdrec', 'mrnn', 'stmvl', 'iim' | default 'cdrec'
            :param n_calls: bayesian parameters, number of calls to the objective function.

            :return : Tuple[dict, Union[Union[int, float, complex], Any]], the best parameters and their corresponding scores.
            """
            # Map the parameter ranges to the algorithm-specific search space
            param_ranges = ALL_ALGO_PARAMS[algorithm]

            # Extract parameter names and their ranges for the selected algorithm
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())

            # Generate all combinations of parameters in the search space
            param_combinations = list(product(*param_values))  # Cartesian product of all parameter values

            # Placeholder for the best parameters and their score
            best_params = None
            best_score = float('inf')  # Assuming we are minimizing the objective function

            def objective(params):
                errors = Imputation.evaluate_params(ground_truth, contamination, params, algorithm)
                return np.mean([errors[metric] for metric in selected_metrics])

            run_count = 0
            # Conduct greedy optimization over parameter combinations
            for params in param_combinations:

                if n_calls is not None and run_count >= n_calls:
                    break

                # Convert params to a dictionary for compatibility
                params_dict = {name: value for name, value in zip(param_names, params)}

                # Calculate the score for the current set of parameters
                score = objective(params_dict)

                # Update the best parameters if the current score is better
                if score < best_score:
                    best_score = score
                    best_params = params_dict

                # Increment the run counter
                run_count += 1

            print(f"Best score: {best_score} with params {best_params}")
            return best_params, best_score

    class Bayesian:

        def optimize(ground_truth, contamination, selected_metrics=["RMSE"], algorithm="cdrec",
                     n_calls=100, n_random_starts=50, acq_func='gp_hedge'):
            """
            Conduct the Bayesian optimization hyperparameter optimization.

            Parameters
            ----------
            :param ground_truth : time series data set to optimize
            :param contamination : time series contaminate to impute
            :param selected_metrics : list of selected metrics to consider for optimization. | default ["RMSE"]
            :param algorithm : imputation algorithm | Valid values: 'cdrec', 'mrnn', 'stmvl', 'iim' | default 'cdrec'
            :param n_calls: bayesian parameters, number of calls to the objective function.
            :param n_random_starts: bayesian parameters, number of initial calls to the objective function, from random points.
            :param acq_func: bayesian parameters, function to minimize over the Gaussian prior (one of 'LCB', 'EI', 'PI', 'gp_hedge') | default gp_hedge

            :return : Tuple[dict, Union[Union[int, float, complex], Any]], the best parameters and their corresponding scores.
            """

            # Adjust the search space for 'cdrec' based on obfuscated_matrix
            if algorithm == 'cdrec':
                max_rank = contamination.shape[1] - 1
                SEARCH_SPACES['cdrec'][0] = Integer(0, min(9, max_rank), name='rank')  # Update the rank range

            # Define the search space
            space = search_spaces[algorithm]

            # Define the objective function (to minimize)
            @use_named_args(space)
            def objective(**params):
                errors = Imputation.evaluate_params(ground_truth, contamination, tuple(params.values()), algorithm)
                return np.mean([errors[metric] for metric in selected_metrics])

            # Conduct Bayesian optimization
            optimizer = skopt.Optimizer(dimensions=space, n_initial_points=n_random_starts, acq_func=acq_func)
            for i in range(n_calls):
                suggested_params = optimizer.ask()
                score = objective(suggested_params)
                optimizer.tell(suggested_params, score)

            # Optimal parameters
            optimal_params = optimizer.Xi[np.argmin(optimizer.yi)]
            optimal_params_dict = {name: value for name, value in zip([dim.name for dim in space], optimal_params)}

            return optimal_params_dict, np.min(optimizer.yi)
