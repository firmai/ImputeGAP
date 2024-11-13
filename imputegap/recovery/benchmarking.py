import time
from imputegap.tools import utils
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries

import os
import matplotlib.pyplot as plt

class Benchmarking:


    def _config_optimization(self, opti_mean, ts_test, scenario, algorithm):
        """
        Configure and execute optimization for selected imputation algorithm and scenario.

        Parameters
        ----------
        opti_mean : float
            Mean parameter for contamination.
        ts_test : TimeSeries
            TimeSeries object containing dataset.
        scenario : str
            Type of contamination scenario (e.g., "mcar", "mp", "blackout").
        algorithm : str
            Imputation algorithm to use.

        Returns
        -------
        BaseImputer
            Configured imputer instance with optimal parameters.
        """

        if scenario == "mcar":
            infected_matrix_opti = ts_test.Contaminate.mcar(ts=ts_test.data, series_impacted=opti_mean,
                                                            missing_rate=opti_mean, use_seed=True, seed=42)
        elif scenario == "mp":
            infected_matrix_opti = ts_test.Contaminate.missing_percentage(ts=ts_test.data, series_impacted=opti_mean,
                                                                          missing_rate=opti_mean)
        else:
            infected_matrix_opti = ts_test.Contaminate.blackout(ts=ts_test.data, missing_rate=opti_mean)

        if algorithm == "cdrec":
            i_opti = Imputation.MatrixCompletion.CDRec(infected_matrix_opti)
        elif algorithm == "stmvl":
            i_opti = Imputation.PatternSearch.STMVL(infected_matrix_opti)
        elif algorithm == "iim":
            i_opti = Imputation.Statistics.IIM(infected_matrix_opti)
        elif algorithm == "mrnn":
            i_opti = Imputation.DeepLearning.MRNN(infected_matrix_opti)
        elif algorithm == "mean":
            i_opti = Imputation.Statistics.MeanImpute(infected_matrix_opti)

        return i_opti




    def generate_reports(self, runs_plots_scores, save_dir="./reports", dataset=""):
        """
        Generate and save a text reports of metrics and timing for each dataset, algorithm, and scenario.

        Parameters
        ----------
        runs_plots_scores : dict
            Dictionary containing scores and timing information for each dataset, scenario, and algorithm.
        save_dir : str, optional
            Directory to save the reports file (default is "./reports").
        dataset : str, optional
            Name of the data for the reports name.

        Returns
        -------
        None

        Notes
        -----
        The reports is saved in a "reports.txt" file in `save_dir`, organized in tabular format.
        """

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "report_"+str(dataset)+".txt")
        with open(save_path, "w") as file:
            file.write("dictionary of results : " + str(runs_plots_scores) + "\n\n")

            # Define header with time columns included
            header = "| dataset_value | algorithm_value | optimizer_value | scenario_value | x_value | RMSE | MAE | MI | CORRELATION | time_contamination | time_optimization | time_imputation |\n"
            file.write(header)

            for dataset, algo_data in runs_plots_scores.items():
                for algorithm, opt_data in algo_data.items():
                    for optimizer, scenario_data in opt_data.items():
                        for scenario, x_data in scenario_data.items():
                            for x, values in x_data.items():
                                metrics = values["scores"]
                                times = values["times"]

                                # Retrieve each timing value, defaulting to None if absent
                                contamination_time = times.get("contamination", None)
                                optimization_time = times.get("optimization", None)
                                imputation_time = times.get("imputation", None)

                                # Create a reports line with timing details
                                line = (
                                    f"| {dataset} | {algorithm} | {optimizer} | {scenario} | {x} "
                                    f"| {metrics.get('RMSE')} | {metrics.get('MAE')} | {metrics.get('MI')} "
                                    f"| {metrics.get('CORRELATION')} | {contamination_time} sec | {optimization_time} sec"
                                    f"| {imputation_time} sec |\n"
                                )
                                file.write(line)

        print("\nReport recorded in", save_path)

    def generate_plots(self, runs_plots_scores, s="M", v="N", save_dir="./reports"):
        """
        Generate and save plots for each metric and scenario based on provided scores.

        Parameters
        ----------
        runs_plots_scores : dict
            Dictionary containing scores and timing information for each dataset, scenario, and algorithm.
        s : str
            display the number of series in graphs
        v : sts
            display the number of values in graphs
        save_dir : str, optional
            Directory to save generated plots (default is "./reports").

        Returns
        -------
        None

        Notes
        -----
        Saves generated plots in `save_dir`, categorized by dataset, scenario, and metric.
        """
        os.makedirs(save_dir, exist_ok=True)

        for dataset, scenario_data in runs_plots_scores.items():
            for scenario, algo_data in scenario_data.items():
                # Iterate over each metric, generating separate plots, including new timing metrics
                for metric in ["RMSE", "MAE", "MI", "CORRELATION", "imputation_time", "optimization_time",
                               "contamination_time"]:
                    plt.figure(figsize=(10, 4))  # Fixed height set by second parameter
                    has_data = False  # Flag to check if any data is added to the plot

                    # Iterate over each algorithm and plot them in the same figure
                    for algorithm, optimizer_data in algo_data.items():
                        x_vals = []
                        y_vals = []
                        for optimizer, x_data in optimizer_data.items():
                            for x, values in x_data.items():
                                # Differentiate between score metrics and timing metrics
                                if metric == "imputation_time" and "imputation" in values["times"]:
                                    x_vals.append(float(x))
                                    y_vals.append(values["times"]["imputation"])
                                elif metric == "optimization_time" and "optimization" in values["times"]:
                                    x_vals.append(float(x))
                                    y_vals.append(values["times"]["optimization"])
                                elif metric == "contamination_time" and "contamination" in values["times"]:
                                    x_vals.append(float(x))
                                    y_vals.append(values["times"]["contamination"])
                                elif metric in values["scores"]:
                                    x_vals.append(float(x))
                                    y_vals.append(values["scores"][metric])

                        # Only plot if there are values to plot
                        if x_vals and y_vals:
                            # Sort x and y values by x for correct spacing
                            sorted_pairs = sorted(zip(x_vals, y_vals))
                            x_vals, y_vals = zip(*sorted_pairs)

                            # Plot each algorithm as a line with scattered points
                            plt.plot(x_vals, y_vals, label=f"{algorithm}")
                            plt.scatter(x_vals, y_vals)
                            has_data = True

                    # Save plot only if there is data to display
                    if has_data:
                        # Set plot titles and labels based on metric
                        title_metric = {
                            "imputation_time": "Imputation Time",
                            "optimization_time": "Optimization Time",
                            "contamination_time": "Contamination Time"
                        }.get(metric, metric)
                        ylabel_metric = {
                            "imputation_time": "Imputation Time (seconds)",
                            "optimization_time": "Optimization Time (seconds)",
                            "contamination_time": "Contamination Time (seconds)"
                        }.get(metric, metric)

                        plt.title(f"{dataset} | {scenario} | {title_metric} | ({s}x{v})")
                        plt.xlabel(f"{scenario} rate of missing values and missing series")
                        plt.ylabel(ylabel_metric)
                        plt.xlim(0.0, 0.85)

                        # Set y-axis limits with padding below 0 for visibility
                        if metric == "imputation_time":
                            plt.ylim(-10, 90)
                        elif metric == "contamination_time":
                            plt.ylim(-0.01, 0.59)
                        elif metric == "MAE":
                            plt.ylim(-0.1, 2.4)
                        elif metric == "MI":
                            plt.ylim(-0.1, 1.85)
                        elif metric == "RMSE":
                            plt.ylim(-0.1, 2.6)
                        elif metric == "CORRELATION":
                            plt.ylim(-0.75, 1.1)

                        # Customize x-axis ticks
                        x_points = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
                        plt.xticks(x_points, [f"{int(tick * 100)}%" for tick in x_points])
                        plt.grid(True, zorder=0)
                        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

                        # Define a unique filename
                        filename = f"{dataset}_{scenario}_{metric}.jpg"
                        filepath = os.path.join(save_dir, filename)

                        # Save the figure
                        plt.savefig(filepath)
                    plt.close()  # Close to avoid memory issues

        print("\nAll plots recorded in", save_dir)


    def comprehensive_evaluation(self, datasets=[], optimizers=[], algorithms=[], scenarios=[], x_axis=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8], save_dir="./reports", already_optimized=False):
        """
        Execute a comprehensive evaluation of imputation algorithms over multiple datasets and scenarios.

        Parameters
        ----------
        datasets : list of str
            List of dataset names to evaluate.
        optimizers : list of dict
            List of optimizers with their configurations.
        algorithms : list of str
            List of imputation algorithms to test.
        scenarios : list of str
            List of contamination scenarios to apply.
        x_axis : list of float
            List of missing rates for contamination.
        save_dir : str, optional
            Directory to save reports and plots (default is "./reports").
        already_optimized : bool, optional
            If True, skip parameter optimization (default is False).

        Returns
        -------
        None

        Notes
        -----
        Runs contamination, imputation, and evaluation, then generates plots and a summary reports.
        """

        print("initialization of the comprehensive evaluation. It can take time...\n")

        for dataset in datasets:

            runs_plots_scores = {}

            limitation_series = 100

            print("1. evaluation launch for", dataset, "========================================================\n\n\n")
            ts_test = TimeSeries()

            header = False
            if dataset == "eeg-reading":
                header = True
            elif dataset == "drift":
                limitation_series = 50

            ts_test.load_timeseries(data=utils.search_path(dataset), max_series=limitation_series, header=header)
            start_time_opti = 0
            end_time_opti = 0

            M, N = ts_test.data.shape

            print("1. normalization of ", dataset, "\n")
            ts_test.normalize()

            for scenario in scenarios:
                print("\t2. contamination of", dataset, "with scenario", scenario, "\n")

                for algorithm in algorithms:
                    has_been_optimized = False
                    print("\t3. algorithm selected", algorithm, "\n")

                    for x in x_axis:
                        print("\t\t4. missing values (series&values) set to", x, "for x_axis\n")

                        start_time_contamination = time.time()  # Record start time
                        if scenario == "mcar":
                            infected_matrix = ts_test.Contaminate.mcar(ts=ts_test.data, series_impacted=x, missing_rate=x, use_seed=True, seed=42)
                        elif scenario == "mp":
                            infected_matrix = ts_test.Contaminate.missing_percentage(ts=ts_test.data, series_impacted=x, missing_rate=x)
                        else:
                            infected_matrix = ts_test.Contaminate.blackout(ts=ts_test.data, missing_rate=x)
                        end_time_contamination = time.time()

                        for optimizer in optimizers:
                            optimizer_gt = {"ground_truth": ts_test.data, **optimizer}

                            if algorithm == "cdrec":
                                algo = Imputation.MatrixCompletion.CDRec(infected_matrix)
                            elif algorithm == "stmvl":
                                algo = Imputation.PatternSearch.STMVL(infected_matrix)
                            elif algorithm == "iim":
                                algo = Imputation.Statistics.IIM(infected_matrix)
                            elif algorithm == "mrnn":
                                algo = Imputation.DeepLearning.MRNN(infected_matrix)
                            elif algorithm == "mean":
                                algo = Imputation.Statistics.MeanImpute(infected_matrix)

                            if not has_been_optimized and not already_optimized and algorithm != "mean":
                                print("\t\t5. AutoML to set the parameters", optimizer, "\n")
                                start_time_opti = time.time()  # Record start time
                                i_opti = self._config_optimization(0.25, ts_test, scenario, algorithm)
                                i_opti.impute(user_defined=False, params=optimizer_gt)
                                utils.save_optimization(optimal_params=i_opti.parameters, algorithm=algorithm, dataset=dataset, optimizer="e")
                                has_been_optimized = True
                                end_time_opti = time.time()

                            if algorithm != "mean":
                                opti_params = utils.load_parameters(query="optimal", algorithm=algorithm, dataset=dataset, optimizer="e")
                                print("\t\t6. imputation", algorithm, "with optimal parameters", *opti_params)

                            else:
                                opti_params = None

                            start_time_imputation = time.time()  # Record start time
                            algo.impute(params=opti_params)
                            end_time_imputation = time.time()

                            algo.score(raw_matrix=ts_test.data, imputed_matrix=algo.imputed_matrix)

                            time_contamination = end_time_contamination - start_time_contamination
                            time_opti = end_time_opti - start_time_opti
                            time_imputation = end_time_imputation - start_time_imputation

                            dic_timing = {"contamination": time_contamination, "optimization": time_opti,
                                          "imputation": time_imputation}

                            dataset_s = dataset
                            if "-" in dataset:
                                dataset_s = dataset.replace("-", "")

                            optimizer_value = optimizer.get('optimizer')  # or optimizer['optimizer']

                            runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(str(scenario), {}).setdefault(
                                str(algorithm), {}).setdefault(str(optimizer_value), {})[str(x)] = {
                                "scores": algo.metrics,
                                "times": dic_timing
                            }

                            print("\t\truns_plots_scores", runs_plots_scores)

            print("\truns_plots_scores : ", runs_plots_scores)
            self.generate_plots(runs_plots_scores=runs_plots_scores, s=str(M), v=str(N), save_dir=save_dir)
            self.generate_reports(runs_plots_scores, save_dir, dataset)

            print("======================================================================================\n\n\n\n\n\n")

        return runs_plots_scores
