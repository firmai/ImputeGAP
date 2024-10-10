import os
import toml
import time
from itertools import product
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools.algorithm_parameters import SEARCH_SPACES, ALL_ALGO_PARAMS, PARAM_NAMES, SEARCH_SPACES_PSO

# PSO IMPORT
from functools import partial
import pyswarms as ps

# BAYESIAN IMPORT
import skopt
from skopt.utils import use_named_args
from skopt.space import Integer

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
            Conduct the Greedy for hyperparameters.

            Parameters
            ----------
            :param ground_truth : time series data set to optimize
            :param contamination : time series contaminate to impute
            :param selected_metrics : list of selected metrics to consider for optimization. | default ["RMSE"]
            :param algorithm : imputation algorithm | Valid values: 'cdrec', 'mrnn', 'stmvl', 'iim' | default 'cdrec'
            :param n_calls: bayesian parameters, number of calls to the objective function.

            :return : Tuple[dict, Union[Union[int, float, complex], Any]], the best parameters and their corresponding scores.
            """
            start_time = time.time()  # Record start time

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

            end_time = time.time()
            print(f"\n\t\t> logs, optimization greedy - Execution Time: {(end_time - start_time):.4f} seconds\n")

            return best_params, best_score

    class Bayesian:

        def optimize(ground_truth, contamination, selected_metrics=["RMSE"], algorithm="cdrec", n_calls=100, n_random_starts=50, acq_func='gp_hedge'):
            """
            Conduct the Bayesian optimization for hyperparameters.

            Parameters
            ----------
            :param ground_truth : time series data set to optimize
            :param contamination : time series contaminate to impute
            :param selected_metrics : list of selected metrics to consider for optimization. | default ["RMSE"]
            :param algorithm : imputation algorithm | Valid values: 'cdrec', 'mrnn', 'stmvl', 'iim' | default 'cdrec'
            :param n_calls: bayesian parameters, number of calls to the objective function.
            :param n_random_starts: bayesian parameters, number of initial calls to the objective function, from random points.
            :param acq_func: bayesian parameters, function to minimize over the Gaussian prior (one of 'LCB', 'EI', 'PI', 'gp_hedge') | default gp_hedge

            :return : Tuple[dict, Union[Union[int, float, complex], Any]], the best parameters and their corresponding scores, Cost of the optimization
            """
            start_time = time.time()  # Record start time

            search_spaces = SEARCH_SPACES

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

            end_time = time.time()
            print(f"\n\t\t> logs, optimization bayesian - Execution Time: {(end_time - start_time):.4f} seconds\n")

            return optimal_params_dict, np.min(optimizer.yi)

    class ParticleSwarm:

        def _format_params(self, particle_params, algorithm):
            if algorithm == 'cdrec':
                particle_params = [int(particle_params[0]), particle_params[1], int(particle_params[2])]
            if algorithm == 'iim':
                particle_params = [int(particle_params[0])]
            elif algorithm == 'mrnn':
                particle_params = [int(particle_params[0]), particle_params[1], int(particle_params[2])]
            elif algorithm == 'stmvl':
                particle_params = [int(particle_params[0]), particle_params[1], int(particle_params[2])]

            return particle_params

        def _objective(self, ground_truth, contamination, algorithm, selected_metrics, params):
            n_particles = params.shape[0]  # Get the number of particles
            # Initialize array to hold the errors for each particle
            errors_for_all_particles = np.zeros(n_particles)


            for i in range(n_particles):  # Iterate over each particle
                particle_params = self._format_params(params[i], algorithm)  # Get the parameters for this particle
                errors = Imputation.evaluate_params(ground_truth, contamination, tuple(particle_params), algorithm)
                errors_for_all_particles[i] = np.mean([errors[metric] for metric in selected_metrics])
            return errors_for_all_particles

        def optimize(self, ground_truth, contamination, selected_metrics, algorithm, n_particles, c1, c2, w, iterations, n_processes):
            """
            Conduct the Particle swarm optimization for hyperparameters.

            Parameters
            ----------
            :param ground_truth : time series data set to optimize
            :param contamination : time series contaminate to impute
            :param selected_metrics : list of selected metrics to consider for optimization. | default ["RMSE"]
            :param algorithm : imputation algorithm | Valid values: 'cdrec', 'mrnn', 'stmvl', 'iim' | default 'cdrec'
            :param n_particles: pso parameters, number of particles used
            :param c1: pso parameters, c1 option value
            :param c2: pso parameters, c2 option value
            :param w: pso parameters, w option value
            :param iterations: pso parameters, number of iterations for the optimization
            :param n_processes: pso parameters, number of process during the optimization

            :return : Tuple[dict, Union[Union[int, float, complex], Any]], the best parameters and their corresponding scores, Cost of the optimization
            """
            start_time = time.time()  # Record start time

            # Define the search space
            search_space = SEARCH_SPACES_PSO


            if algorithm == 'cdrec':
                max_rank = contamination.shape[1] - 1
                search_space['cdrec'][0] = (search_space['cdrec'][0][0], min(search_space['cdrec'][0][1], max_rank))

            # Select the correct search space based on the algorithm
            bounds = search_space[algorithm]

            # Convert search space to PSO-friendly format (two lists: one for min and one for max values for each parameter)
            lower_bounds, upper_bounds = zip(*bounds)
            bounds = (np.array(lower_bounds), np.array(upper_bounds))

            # Call instance of PSO
            optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=len(bounds[0]), options={'c1': c1, 'c2': c2, 'w': w}, bounds=bounds)

            # Perform optimization
            objective_with_args = partial(self._objective, ground_truth, contamination, algorithm, selected_metrics)
            cost, pos = optimizer.optimize(objective_with_args, iters=iterations, n_processes=n_processes)

            param_names = PARAM_NAMES

            optimal_params = self._format_params(pos, algorithm)
            optimal_params_dict = {param_name: value for param_name, value in zip(param_names[algorithm], optimal_params)}


            end_time = time.time()
            print(f"\n\t\t> logs, optimization pso - Execution Time: {(end_time - start_time):.4f} seconds\n")

            return optimal_params_dict, cost
