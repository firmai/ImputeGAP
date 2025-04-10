import datetime
import os
import math
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries



class Benchmark:
    """
    A class to evaluate the performance of imputation algorithms through benchmarking across datasets and patterns.

    Methods
    -------
    average_runs_by_names(self, data):
        Average the results of all runs depending on the dataset.
    avg_results():
        Calculate average metrics (e.g., RMSE) across multiple datasets and algorithm runs.
    generate_heatmap():
        Generate and save a heatmap visualization of RMSE scores for datasets and algorithms.
    generate_reports_txt():
        Create detailed text-based reports summarizing metrics and timing results for all evaluations.
    generate_reports_excel():
        Create detailed excel-based reports summarizing metrics and timing results for all evaluations.
    generate_plots():
        Visualize metrics (e.g., RMSE, MAE) and timing (e.g., imputation, optimization) across patterns and datasets.
    eval():
        Perform a complete benchmarking pipeline, including contamination, imputation, evaluation, and reporting.

    Example
    -------
    output : {'eegalcohol': {'mcar': {'SoftImpute': {'default_params': {'0.05': {'scores': {'RMSE': 0.4359915238078244, 'MAE': 0.3725965559420608, 'MI': 1.4169232775678364, 'CORRELATION': 0.9530448037164908, 'runtime': 0.2811245918273926, 'runtime_log_scale': -0.5511011621200811}}, '0.1': {'scores': {'RMSE': 0.3665001858394363, 'MAE': 0.2989983612840734, 'MI': 0.9078918430616858, 'CORRELATION': 0.9049909722894052, 'runtime': 0.1305081844329834, 'runtime_log_scale': -0.8843622519837359}}, '0.2': {'scores': {'RMSE': 0.39833006221984, 'MAE': 0.30824644022807457, 'MI': 0.8483406827418594, 'CORRELATION': 0.9161465703422209, 'runtime': 0.1445169448852539, 'runtime_log_scale': -0.8400812280720497}}, '0.4': {'scores': {'RMSE': 0.435591016228979, 'MAE': 0.3335144215651955, 'MI': 0.7286325588353783, 'CORRELATION': 0.9021032587324183, 'runtime': 0.306765079498291, 'runtime_log_scale': -0.5131940796785996}}, '0.6': {'scores': {'RMSE': 0.4500113661547204, 'MAE': 0.338085865703361, 'MI': 0.6481512576687939, 'CORRELATION': 0.8893263437029546, 'runtime': 0.10979938507080078, 'runtime_log_scale': -0.9594000921364352}}, '0.8': {'scores': {'RMSE': 0.46554422402146944, 'MAE': 0.3508926604243284, 'MI': 0.6150677913271478, 'CORRELATION': 0.8791443563129441, 'runtime': 0.12360215187072754, 'runtime_log_scale': -0.907973968264617}}}}, 'KNNImpute': {'default_params': {'0.05': {'scores': {'RMSE': 0.24102595399583507, 'MAE': 0.18984832836399548, 'MI': 1.547782862758484, 'CORRELATION': 0.9810468571465141, 'runtime': 0.04365086555480957, 'runtime_log_scale': -1.3600071402298746}}, '0.1': {'scores': {'RMSE': 0.28890851809839135, 'MAE': 0.22998623733608023, 'MI': 0.9934511613817691, 'CORRELATION': 0.942944831550703, 'runtime': 0.02493762969970703, 'runtime_log_scale': -1.6031448282134637}}, '0.2': {'scores': {'RMSE': 0.32523842533021824, 'MAE': 0.2398150375225743, 'MI': 1.0141220941333857, 'CORRELATION': 0.9441733384102147, 'runtime': 0.01948857307434082, 'runtime_log_scale': -1.7102199581438617}}, '0.4': {'scores': {'RMSE': 0.33824584758611187, 'MAE': 0.2509197576351218, 'MI': 0.9534836631617412, 'CORRELATION': 0.9418540692188737, 'runtime': 0.04377126693725586, 'runtime_log_scale': -1.3588108827701144}}, '0.6': {'scores': {'RMSE': 0.3656435952078159, 'MAE': 0.26630724595251076, 'MI': 0.7933983583413302, 'CORRELATION': 0.9285706343976159, 'runtime': 0.10764479637145996, 'runtime_log_scale': -0.9680069594473933}}, '0.8': {'scores': {'RMSE': 0.49851932645867886, 'MAE': 0.3727407177987301, 'MI': 0.5872198065951101, 'CORRELATION': 0.8588039019214768, 'runtime': 0.1279888153076172, 'runtime_log_scale': -0.8928279808397801}}}}}}}
    """

    def __init__(self):
        """
        Initialize the Benchmark object.
        """
        self.list_results = None
        self.aggregate_results = None
        self.heatmap = None
        self.plots = None


    def _benchmark_exception(self, data, algorithm, pattern, x):
        """
        Check whether a specific algorithm-pattern combination should be excluded from benchmarking.

        This function flags exceptions where benchmarking is not appropriate or known to fail,
        based on the algorithm name, the missingness pattern, and the missingness rate `x`.

        Parameters
        ----------
        data : numpy matrix
            Matrix of data with nan values
        algorithm : str
            Name of the imputation algorithm (e.g., 'DEEPMVI', 'PRISTI').
        pattern : str
            Missing data pattern (e.g., 'MCAR', 'ALIGNED').
        x : float
            Proportion of missing values in the data (between 0 and 1).

        Returns
        -------
        bool
            True if the benchmark should be skipped for the given configuration, False otherwise.

        Rules
        -----
            - For DeepMVI with MCAR pattern and x > 0.6, skip benchmarking.
            - For PRISTI, always skip benchmarking.
        """

        if algorithm.upper() == 'DEEPMVI' or algorithm.upper() == 'DEEP_MVI':
            if pattern.lower() == "mcar" or pattern.lower() == "missing_completely_at_random":
                if x > 0.6:
                    print(f"\n(BENCH) The imputation algorithm {algorithm} is not compatible with this configuration {pattern} with missingness rate more than 0.6.")
                    return True
            if pattern.lower() == "mp" or pattern.lower() == "aligned":
                if x < 0.15:
                    print(f"\n(BENCH) The imputation algorithm {algorithm} is not compatible with this configuration {pattern} with missingness rate less then 0.15.")
                    return True

        if algorithm.upper() == 'MPIN':
            print(f"\n(BENCH) The imputation algorithm {algorithm} is not compatible with this setup.")
            return True

        return False

    def _config_optimization(self, opti_mean, ts_test, pattern, algorithm, block_size_mcar):
        """
        Configure and execute optimization for selected imputation algorithm and pattern.

        Parameters
        ----------
        opti_mean : float
            Mean parameter for contamination.
        ts_test : TimeSeries
            TimeSeries object containing dataset.
        pattern : str
            Type of contamination pattern (e.g., "mcar", "mp", "blackout", "disjoint", "overlap", "gaussian").
        algorithm : str
            Imputation algorithm to use.
        block_size_mcar : int
            Size of blocks removed in MCAR

        Returns
        -------
        BaseImputer
            Configured imputer instance with optimal parameters.
        """

        incomp_data = utils.config_contamination(ts=ts_test, pattern=pattern, dataset_rate=opti_mean, series_rate=opti_mean, block_size=block_size_mcar)
        imputer = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=algorithm)

        return imputer

    def average_runs_by_names(self, data):
        """
        Average the results of all runs depending on the dataset

        Parameters
        ----------
        data : list
            list of dictionary containing the results of the benchmark runs.

        Returns
        -------
        list
            list of dictionary containing the results of the benchmark runs averaged by datasets.
        """
        results_avg, all_names = [], []

        # Extract dataset names
        for dictionary in data:
            all_keys = list(dictionary.keys())
            dataset_name = all_keys[0]
            all_names.append(dataset_name)

        # Get unique dataset names
        unique_names = sorted(set(all_names))

        # Initialize and populate the split matrix
        split = [[0 for _ in range(all_names.count(name))] for name in unique_names]
        for i, name in enumerate(unique_names):
            x = 0
            for y, match in enumerate(all_names):
                if name == match:
                    split[i][x] = data[y]
                    x += 1

        # Iterate over the split matrix to calculate averages
        for datasets in split:
            tmp = [dataset for dataset in datasets if dataset != 0]
            merged_dict = {}
            count = len(tmp)

            # Process and calculate averages
            for dataset in tmp:
                for outer_key, outer_value in dataset.items():
                    for middle_key, middle_value in outer_value.items():
                        for mean_key, mean_value in middle_value.items():
                            for method_key, method_value in mean_value.items():
                                for level_key, level_value in method_value.items():
                                    # Initialize scores and times if not already initialized
                                    merger = merged_dict.setdefault(outer_key, {}
                                                                    ).setdefault(middle_key, {}).setdefault(mean_key, {}
                                                                                                            ).setdefault(
                                        method_key, {}).setdefault(level_key, {"scores": {}})

                                    # Add scores and times
                                    for score_key, v in level_value["scores"].items():
                                        if v is None :
                                            v = 0
                                        merger["scores"][score_key] = (merger["scores"].get(score_key, 0) + v / count)

            results_avg.append(merged_dict)

        return results_avg

    def avg_results(self, *datasets, metric="RMSE"):
        """
        Calculate the average of all metrics and times across multiple datasets.

        Parameters
        ----------
        datasets : dict
            Multiple dataset dictionaries to be averaged.
        metric : str
            Metric to group.

        Returns
        -------
        List
            Matrix with averaged scores and times for all levels, list of algorithms, list of datasets
        """

        # Step 1: Compute average RMSE across runs for each dataset and algorithm
        aggregated_data = {}

        for runs in datasets:
            for dataset, dataset_items in runs.items():
                if dataset not in aggregated_data:
                    aggregated_data[dataset] = {}

                for pattern, pattern_items in dataset_items.items():
                    for algo, algo_data in pattern_items.items():
                        if algo not in aggregated_data[dataset]:
                            aggregated_data[dataset][algo] = []

                        for missing_values, missing_values_item in algo_data.items():
                            for param, param_data in missing_values_item.items():
                                rmse = param_data["scores"][metric]
                                aggregated_data[dataset][algo].append(rmse)

        # Step 2: Compute averages using NumPy
        average_rmse_matrix = {}
        for dataset, algos in aggregated_data.items():
            average_rmse_matrix[dataset] = {}
            for algo, rmse_values in algos.items():
                rmse_array = np.array(rmse_values)
                avg_rmse = np.mean(rmse_array)
                average_rmse_matrix[dataset][algo] = avg_rmse

        # Step 3: Create a matrix representation of datasets and algorithms
        datasets_list = list(average_rmse_matrix.keys())
        algorithms = {algo for algos in average_rmse_matrix.values() for algo in algos}
        algorithms_list = sorted(algorithms)

        # Prepare a NumPy matrix
        comprehensive_matrix = np.zeros((len(datasets_list), len(algorithms_list)))

        for i, dataset in enumerate(datasets_list):
            for j, algo in enumerate(algorithms_list):
                comprehensive_matrix[i, j] = average_rmse_matrix[dataset].get(algo, np.nan)

        return comprehensive_matrix, algorithms_list, datasets_list

    def generate_heatmap(self, scores_list, algos, sets, metric="RMSE", save_dir="./reports", display=True):
        """
        Generate and save RMSE matrix in HD quality.

        Parameters
        ----------
        scores_list : np.ndarray
            2D numpy array containing RMSE values.
        algos : list of str
            List of algorithm names (columns of the heatmap).
        sets : list of str
            List of dataset names (rows of the heatmap).
        metric : str, optional
            metric to extract
        save_dir : str, optional
            Directory to save the generated plot (default is "./reports").
        display : bool, optional
            Display or not the plot

        Returns
        -------
        Bool
            True if the matrix has been generated
        """
        save_dir = save_dir + "/_heatmaps/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        nbr_algorithms = len(algos)
        nbr_datasets= len(sets)

        cell_size = 4.0
        x_size = cell_size*nbr_algorithms
        y_size = cell_size*nbr_datasets

        fig, ax = plt.subplots(figsize=(x_size, y_size))
        fig.canvas.manager.set_window_title("benchmark heatmap, " + metric)
        cmap = plt.cm.Greys

        if metric == "RMSE":
            norm = plt.Normalize(vmin=0, vmax=2)
        elif metric == "CORRELATION":
            norm = plt.Normalize(vmin=-2, vmax=2)
        elif metric.lower() == "runtime_linear_scale":
            norm = plt.Normalize(vmin=0, vmax=1000)
        elif metric.lower() == "runtime_log_scale":
            norm = plt.Normalize(vmin=-2, vmax=10)
        else:
            norm = plt.Normalize(vmin=0, vmax=2)

        # Create the heatmap
        heatmap = ax.imshow(scores_list, cmap=cmap, norm=norm, aspect='auto')

        # Add color bar for reference
        cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical')
        cbar.set_label(metric, rotation=270, labelpad=15)

        # Set the tick labels
        ax.set_xticks(np.arange(nbr_algorithms))
        ax.set_xticklabels(algos)
        ax.set_yticks(np.arange(nbr_datasets))
        ax.set_yticklabels(sets)

        # Add titles and labels
        ax.set_title('ImputeGAP Algorithms Comparison')
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Datasets')

        # Show values on the heatmap
        for i in range(len(sets)):
            for j in range(len(algos)):
                ax.text(j, i, f"{scores_list[i, j]:.2f}",
                        ha='center', va='center',
                        color="black" if scores_list[i, j] < 1 else "white")  # for visibility

        filename = "benchmarking_"+ metric.lower()+ ".jpg"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')  # Save in HD with tight layout

        # Show the plot
        if display:
            plt.tight_layout()
            plt.show()
            plt.close()

        self.heatmap = plt

        return True

    def generate_reports_txt(self, runs_plots_scores, save_dir="./reports", dataset="", metrics=["RMSE"], run=-1, rt=0, verbose=True):
        """
        Generate and save a text report of metrics and timing for each dataset, algorithm, and pattern.

        Parameters
        ----------
        runs_plots_scores : dict
            Dictionary containing scores and timing information for each dataset, pattern, and algorithm.
        save_dir : str, optional
            Directory to save the reports file (default is "./reports").
        dataset : str, optional
            Name of the data for the report name.
        metrics : str, optional
            List of metrics asked for in the report.
        run : int, optional
            Number of the run.
        rt : float, optional
            Total time of the run.
        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        None

        Notes
        -----
        The report is saved in a "report.txt" file in `save_dir`, organized in sections with headers and results.
        """
        os.makedirs(save_dir, exist_ok=True)


        if "RMSE" not in metrics:
            to_call = [metrics[0], "runtime"]
        else:
            to_call = ["RMSE", "runtime"]

        new_metrics = np.copy(metrics)

        if metrics is None:
            new_metrics = utils.list_of_metrics()
        else:
            if "runtime" not in new_metrics:
                new_metrics = np.append(new_metrics, "runtime")
            if "runtime_log_scale" not in new_metrics:
                new_metrics = np.append(new_metrics, "runtime_log_scale")

        opt = None
        for dataset, patterns_items in runs_plots_scores.items():
            for pattern, algorithm_items in patterns_items.items():
                for algorithm, optimizer_items in algorithm_items.items():
                    for optimizer, x_data_items in optimizer_items.items():
                        opt = optimizer
                        break

        list_of_patterns = []
        for dataset, patterns_items in runs_plots_scores.items():
            for pattern, algorithm_items in patterns_items.items():
                list_of_patterns.append(pattern)
                new_dir = save_dir + "/" + pattern.lower() + "/"
                os.makedirs(new_dir, exist_ok=True)

                save_path = os.path.join(new_dir, f"report_{pattern}_{dataset}.txt")
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(save_path, "w") as file:
                    file.write(f"Report for Dataset: {dataset}\n")
                    file.write(f"Generated on: {current_time}\n")
                    file.write(f"Total runtime: {rt} (sec)\n")
                    if run >= 0:
                        file.write(f"Run number: {run}\n")
                    file.write("=" * 120 + "\n\n")

                    for metric in new_metrics:
                        if metric == "runtime_log_scale":
                            break

                        file.write(f"\n{dataset}: {{{pattern}, {metric}, {opt}}}")

                        # Collect all algorithms and scores by rate
                        rate_to_scores = defaultdict(dict)
                        all_algorithms = set()

                        for algorithm, optimizer_items in algorithm_items.items():
                            for optimizer, x_data_items in optimizer_items.items():
                                for x, values in x_data_items.items():
                                    score = values.get("scores", {}).get(metric, None)
                                    if score is not None:
                                        rate_to_scores[x][algorithm] = f"{score:.10f}"
                                        all_algorithms.add(algorithm)

                        all_algorithms = sorted(all_algorithms)
                        headers = ["Rate"] + list(all_algorithms)
                        column_widths = [5] + [18] * len(all_algorithms)

                        # Header and separator rows
                        header_row = "".join(f" {header:^{width}} " for header, width in zip(headers, column_widths))
                        separator_row = "" + "".join(f"{'' * (width + 2)}" for width in column_widths) + ""

                        file.write(f"{separator_row}\n")
                        file.write(f"{header_row}\n")
                        file.write(f"{separator_row}\n")

                        if metric in to_call and verbose:
                            print(f"\n{dataset}: {{{pattern}, {metric}, {opt}}}")
                            print(separator_row)
                            print(f"{header_row}")
                            print(separator_row)

                        # Write each row
                        for rate in sorted(rate_to_scores.keys()):
                            row_values = [rate] + [rate_to_scores[rate].get(algo, "") for algo in all_algorithms]
                            row = "".join(f" {val:^{width}} " for val, width in zip(row_values, column_widths))
                            file.write(f"{row}\n")
                            if metric in to_call and verbose:
                                print(f"{row}")

                        file.write(f"{separator_row}\n\n")
                        if metric in to_call and verbose:
                            print(separator_row + "\n")

                    file.write("Dictionary of Results:\n")
                    file.write(str(runs_plots_scores) + "\n")


    def generate_reports_excel(self, runs_plots_scores, save_dir="./reports", dataset="", run=-1, verbose=True):
        """
        Generate and save an Excel-like text report of metrics and timing for each dataset, algorithm, and pattern.

        Parameters
        ----------
        runs_plots_scores : dict
            Dictionary containing scores and timing information for each dataset, pattern, and algorithm.
        save_dir : str, optional
            Directory to save the Excel-like file (default is "./reports").
        dataset : str, optional
            Name of the data for the Excel-like file name.
        run : int, optional
            Number of the run
        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        None
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"report_{dataset}.xlsx")

        # Create an Excel workbook
        workbook = xlsxwriter.Workbook(save_path)

        # Add a summary sheet with the header, creation date, dictionary content, and links to other sheets
        summary_sheet = workbook.add_worksheet("Summary")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_sheet.set_column(0, 1, 50)

        # Title and header
        summary_sheet.write(0, 0, "ImputeGAP, A library of Imputation Techniques for Time Series Data")
        summary_sheet.write(2, 0, "Report for Dataset")
        summary_sheet.write(2, 1, dataset)
        summary_sheet.write(3, 0, "Generated on")
        summary_sheet.write(3, 1, current_time)
        if run >= 0:
            summary_sheet.write(4, 0, "Run Number")
            summary_sheet.write(4, 1, run)

        # Add links to metric sheets
        row = 6
        summary_sheet.write(row, 0, "Metric Sheets:")
        row += 1
        metrics = {
            "RMSE": "Root Mean Square Error - Measures the average magnitude of error.",
            "MAE": "Mean Absolute Error - Measures the average absolute error.",
            "MI": "Mutual Information - Indicates dependency between variables.",
            "CORRELATION": "Correlation Coefficient - Indicates linear relationship between variables."
        }
        for metric in metrics.keys():
            summary_sheet.write_url(row, 0, f"internal:'{metric}'!A1", string=f"Go to {metric} Sheet")
            row += 1

        # Write the dictionary content
        summary_sheet.write(row + 1, 0, "Dictionary of Results")
        row += 2

        for key, value in runs_plots_scores.items():
            summary_sheet.write(row, 0, str(key))
            summary_sheet.write(row, 1, str(value))
            row += 1

        for metric, description in metrics.items():
            # Create a worksheet for each metric
            worksheet = workbook.add_worksheet(metric)

            # Write the metric description at the top and add IMPUTEGAP header
            worksheet.write(0, 0, "ImputeGAP, A library of Imputation Techniques for Time Series Data")
            worksheet.write(2, 0, f"{metric}: {description}")

            # Define consistent column headers and widths
            headers = ["Dataset", "Algorithm", "Optimizer", "Pattern", "X Value", metric]
            column_widths = [15, 15, 15, 15, 12, 20]  # Adjust widths for Excel

            # Write the headers
            for col, (header, width) in enumerate(zip(headers, column_widths)):
                worksheet.set_column(col, col, width)
                worksheet.write(3, col, header)

            # Populate the data
            row = 4
            for dataset, algo_items in runs_plots_scores.items():
                for algorithm, optimizer_items in algo_items.items():
                    for optimizer, pattern_data in optimizer_items.items():
                        for pattern, x_data_items in pattern_data.items():
                            for x, values in x_data_items.items():
                                value = values.get("scores", {}).get(metric, None)
                                if value is not None:
                                    value = f"{value:.10f}"
                                    data = [dataset, algorithm, optimizer, pattern, str(x), value]
                                    for col, cell_value in enumerate(data):
                                        worksheet.write(row, col, cell_value)
                                    row += 1

        # Close the workbook
        workbook.close()


    def generate_plots(self, runs_plots_scores, ticks,  metrics=None, subplot=False, y_size=4, title=None, save_dir="./reports",display=False, verbose=True):
        """
        Generate and save plots for each metric and pattern based on provided scores.

        Parameters
        ----------
        runs_plots_scores : dict
            Dictionary containing scores and timing information for each dataset, pattern, and algorithm.
        ticks : list of float
            List of missing rates for contamination.
        metrics : list of string
            List of metrics used
        subplot : bool, optional
            If True, generates a single figure with subplots for all metrics (default is False).
        y_size : int, optional
            Default size of the graph (default is 4).
        title : str, optional
            Title of the graph (default is "imputegap benchmark").
        save_dir : str, optional
            Directory to save generated plots (default is "./reports").
        display : bool, optional
            Display or not the plots (default is False).
        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        None

        Notes
        -----
        Saves generated plots in `save_dir`, categorized by dataset, pattern, and metric.
        """
        os.makedirs(save_dir, exist_ok=True)

        new_metrics = np.copy(metrics)
        new_plots = 0

        if metrics is None:
            new_metrics = utils.list_of_metrics()
        else:
            if "runtime" not in new_metrics:
                new_metrics = np.append(new_metrics, "runtime")
                new_plots = new_plots + 1
            if "runtime_log_scale" not in new_metrics:
                new_plots = new_plots + 1
                new_metrics = np.append(new_metrics, "runtime_log_scale")

        n_rows = int((len(new_metrics)+new_plots)/2)

        x_size, title_flag = 16, title

        for dataset, pattern_items in runs_plots_scores.items():
            for pattern, algo_items in pattern_items.items():
                if subplot:
                    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(x_size*1.90, y_size*2.90))  # Adjusted figsize
                    if title_flag is None:
                        title = dataset + " : " + pattern + ", benchmark analysis"
                    fig.canvas.manager.set_window_title(title)
                    axes = axes.ravel()  # Flatten the 2D array of axes to a 1D array

                # Iterate over each metric, generating separate plots, including new timing metrics
                for i, metric in enumerate(new_metrics):
                    if subplot:
                        if i < len(axes):
                            ax = axes[i]
                        else:
                            break  # Prevent index out of bounds if metrics exceed subplot slots
                    else:
                        plt.figure(figsize=(x_size, y_size))
                        ax = plt.gca()

                    has_data = False  # Flag to check if any data is added to the plot
                    max_y, min_y = -99999, 99999
                    for algorithm, optimizer_items in algo_items.items():
                        x_vals = []
                        y_vals = []
                        for optimizer, x_data in optimizer_items.items():
                            for x, values in x_data.items():
                                if metric in values["scores"]:
                                    x_vals.append(float(x))
                                    y_vals.append(values["scores"][metric])

                        if x_vals and y_vals:
                            sorted_pairs = sorted(zip(x_vals, y_vals))
                            x_vals, y_vals = zip(*sorted_pairs)

                            # Plot each algorithm as a line with scattered points
                            ax.plot(x_vals, y_vals, label=f"{algorithm}")
                            ax.scatter(x_vals, y_vals)
                            has_data = True


                            if min_y > min(y_vals):
                                min_y = min(y_vals)
                            if max_y < max(y_vals):
                                max_y = max(y_vals)

                    # Save plot only if there is data to display
                    if has_data:
                        ylabel_metric = {
                            "runtime": "Runtime (sec)",
                            "runtime_log_scale": "Runtime (sec)",
                        }.get(metric, metric)

                        ax.set_title(metric)
                        ax.set_xlabel("Rates")
                        ax.set_ylabel(ylabel_metric)
                        ax.set_xlim(0.0, 0.85)

                        if metric == "RMSE" or metric == "MAE":
                            if min_y < 0:
                                min_y = 0
                            if max_y > 3:
                                max_y = 3
                        elif metric == "CORRELATION":
                            if min_y < -1:
                                min_y = -1
                            if max_y > 1:
                                max_y = 1
                        elif metric == "MI":
                            if min_y < 0:
                                min_y = 0
                            if max_y > 2:
                                max_y = 2
                        elif metric == "runtime":
                            if min_y < 0:
                                min_y = 0
                            if max_y > 10:
                                max_y = 10
                        elif metric == "runtime_log_scale":
                            if min_y < -5:
                                min_y = -5
                            if max_y > 5:
                                max_y = 5

                        diff = (max_y - min_y)
                        y_padding = 0.15*diff

                        if y_padding is None or y_padding == 0:
                            y_padding = 1

                        ax.set_ylim(min_y - y_padding, max_y + y_padding)

                        # Set y-axis limits with padding below 0 for visibility
                        if metric == "runtime":
                            ax.set_title("Runtime (linear scale)")
                        elif metric == "runtime_log_scale":
                            ax.set_title("Runtime (log scale)")
                        elif metric == "CORRELATION":
                            ax.set_title("Pearson Correlation")

                        # Customize x-axis ticks
                        ax.set_xticks(ticks)
                        ax.set_xticklabels([f"{int(tick * 100)}%" for tick in ticks])
                        ax.grid(True, zorder=0)
                        ax.legend(loc='upper left', fontsize=7, frameon=True, fancybox=True, framealpha=0.8)

                    if not subplot:
                        filename = f"{dataset}_{pattern}_{optimizer}_{metric}.jpg"

                        new_dir = save_dir + "/" + pattern
                        os.makedirs(new_dir, exist_ok=True)

                        filepath = os.path.join(new_dir, filename)
                        plt.savefig(filepath)
                        if not display:
                            plt.close()

                if subplot:
                    plt.tight_layout()
                    new_dir = save_dir + "/" + pattern
                    os.makedirs(new_dir, exist_ok=True)
                    filename = f"{dataset}_{pattern}_metrics_subplot.jpg"
                    filepath = os.path.join(new_dir, filename)
                    plt.savefig(filepath)

                    if display:
                        plt.show()
                    else:
                        plt.close()

        self.plots = plt

    def eval(self, algorithms=["cdrec"], datasets=["eeg-alcohol"], patterns=["mcar"], x_axis=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8], optimizers=["default_params"], metrics=["*"], save_dir="./imputegap_assets/benchmark", runs=1, normalizer="z_score", nbr_series=None, nbr_vals=None, verbose=False):
        """
        Execute a comprehensive evaluation of imputation algorithms over multiple datasets and patterns.

        Parameters
        ----------
        algorithms : list of str
            List of imputation algorithms to test.
        datasets : list of str
            List of dataset names to evaluate.
        patterns : list of str
            List of contamination patterns to apply.
        x_axis : list of float
            List of missing rates for contamination.
        optimizers : list
            List of optimizers with their configurations.
        metrics : list of str
            List of metrics for evaluation.
        save_dir : str, optional
            Directory to save reports and plots (default is "./reports").
        runs : int, optional
            Number of executions with a view to averaging them
        normalizer : str, optional
            Normalizer to pre-process the data (default is "z_score").
        nbr_series : int, optional
            Number of series to take inside the dataset (default is None > all series).
        nbr_vals : int, optional
            Number of values to take inside the series (default is None > all values).
        verbose : bool, optional
            Whether to display the contamination information (default is False).

        Returns
        -------
        List
            List of all runs results, matrix with averaged scores and times for all levels

        Notes
        -----
        Runs contamination, imputation, and evaluation, then generates plots and a summary reports.
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        run_storage = []
        not_optimized = ["none"]
        mean_group = ["mean", "MeanImpute", "min", "MinImpute", "zero", "ZeroImpute", "MeanImputeBySeries"]

        if "*" in metrics or "all" in metrics:
            metrics = utils.list_of_metrics()
        if "*" in metrics or "all" in algorithms:
            all_algs = utils.list_of_algorithms()
            algorithms = [item for item in all_algs if item.upper() != "MPIN"]

        benchmark_time = time.time()
        for i_run in range(0, abs(runs)):
            for dataset in datasets:
                runs_plots_scores = {}
                block_size_mcar = 10
                y_p_size = max(4, len(algorithms)*0.275)

                if verbose:
                    print("\n1. evaluation launch for", dataset, "\n")
                ts_test = TimeSeries()

                header = False
                if dataset == "eeg-reading":
                    header = True

                ts_test.load_series(data=utils.search_path(dataset), nbr_series=nbr_series, nbr_val=nbr_vals, header=header)

                M, N = ts_test.data.shape

                if N < 250:
                    block_size_mcar = 2

                if normalizer in utils.list_of_normalizers():
                    ts_test.normalize(verbose=verbose)

                for pattern in patterns:
                    if verbose:
                        print("\n2. contamination of", dataset, "with pattern", pattern, "\n")

                    for algorithm in algorithms:
                        has_been_optimized = False
                        if verbose:
                            print("\n3. algorithm evaluated", algorithm, "with", pattern, "\n")
                        else:
                            print(f"{algorithm} using {pattern}, started at {time.strftime('%Y-%m-%d %H:%M:%S')}.")

                        for incx, x in enumerate(x_axis):
                            if verbose:
                                print("\n4. missing values (series&values) set to", x, "for x_axis\n")

                            incomp_data = utils.config_contamination(ts=ts_test, pattern=pattern, dataset_rate=x, series_rate=x, block_size=block_size_mcar, verbose=verbose)

                            for optimizer in optimizers:
                                algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=algorithm, verbose=verbose)

                                if isinstance(optimizer, dict):
                                    optimizer_gt = {"input_data": ts_test.data, **optimizer}
                                    optimizer_value = optimizer.get('optimizer')  # or optimizer['optimizer']

                                    if not has_been_optimized and algorithm not in mean_group and algorithm not in not_optimized:
                                        if verbose:
                                            print("\n5. AutoML to set the parameters", optimizer, "\n")
                                        i_opti = self._config_optimization(0.25, ts_test, pattern, algorithm, block_size_mcar)
                                        i_opti.impute(user_def=False, params=optimizer_gt)
                                        utils.save_optimization(optimal_params=i_opti.parameters, algorithm=algorithm, dataset=dataset, optimizer="e")

                                        has_been_optimized = True
                                    else:
                                        if verbose:
                                            print("\n5. AutoML already optimized...\n")

                                    if algorithm not in mean_group and algorithm not in not_optimized:
                                        if i_opti.parameters is None:
                                            opti_params = utils.load_parameters(query="optimal", algorithm=algorithm, dataset=dataset, optimizer="e")
                                            if verbose:
                                                print("\n6. imputation", algorithm, "with optimal parameters from files", *opti_params)
                                        else:
                                            opti_params = i_opti.parameters
                                            if verbose:
                                                print("\n6. imputation", algorithm, "with optimal parameters from object", *opti_params)
                                    else:
                                        if verbose:
                                            print("\n5. No AutoML launches without optimal params for", algorithm, "\n")
                                        opti_params = None
                                else:
                                    if verbose:
                                        print("\n5. Default parameters have been set the parameters", optimizer, "for", algorithm, "\n")
                                    optimizer_value = optimizer
                                    opti_params = None

                                start_time_imputation = time.time()

                                if not self._benchmark_exception(incomp_data, algorithm, pattern, x):
                                    algo.impute(params=opti_params)
                                else:
                                    algo.recov_data = incomp_data

                                end_time_imputation = time.time()

                                algo.score(input_data=ts_test.data, recov_data=algo.recov_data, verbose=False)

                                if "*" not in metrics and "all" not in metrics:
                                    algo.metrics = {k: algo.metrics[k] for k in metrics if k in algo.metrics}

                                time_imputation = end_time_imputation - start_time_imputation
                                log_time_imputation = math.log10(time_imputation) if time_imputation > 0 else None

                                algo.metrics["runtime"] = time_imputation
                                algo.metrics["runtime_log_scale"] = log_time_imputation

                                dataset_s = dataset
                                if "-" in dataset:
                                    dataset_s = dataset.replace("-", "")

                                runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(str(pattern), {}).setdefault(str(algorithm), {}).setdefault(str(optimizer_value), {})[str(x)] = {"scores": algo.metrics}

                        print(f"done!\n\n")
                #save_dir_runs = save_dir + "/_details/run_" + str(i_run) + "/" + dataset
                #if verbose:
                #    print("\nruns saved in : ", save_dir_runs)
                #self.generate_plots(runs_plots_scores=runs_plots_scores, ticks=x_axis, metrics=metrics, subplot=True, y_size=y_p_size, save_dir=save_dir_runs, display=False, verbose=verbose)
                #self.generate_plots(runs_plots_scores=runs_plots_scores, ticks=x_axis, metrics=metrics, subplot=False, y_size=y_p_size, save_dir=save_dir_runs, display=False, verbose=verbose)
                #self.generate_reports_txt(runs_plots_scores=runs_plots_scores, save_dir=save_dir_runs, dataset=dataset, metrics=metrics, run=i_run, verbose=verbose)
                #self.generate_reports_excel(runs_plots_scores, save_dir_runs, dataset, i_run, verbose=verbose)
                run_storage.append(runs_plots_scores)

        for x, m in enumerate(reversed(metrics)):
            tag = True if x == (len(metrics)-1) else False

            scores_list, algos, sets = self.avg_results(*run_storage, metric=m)
            _ = self.generate_heatmap(scores_list=scores_list, algos=algos, sets=sets, metric=m, save_dir=save_dir, display=tag)

        run_averaged = self.average_runs_by_names(run_storage)

        benchmark_end = time.time()
        total_time_benchmark = round(benchmark_end - benchmark_time, 4)
        print(f"\n> logs: benchmark - Execution Time: {total_time_benchmark} seconds\n")

        verb = True
        for scores in run_averaged:
            all_keys = list(scores.keys())
            dataset_name = str(all_keys[0])
            save_dir_agg_set = save_dir + "/" + dataset_name

            self.generate_reports_txt(runs_plots_scores=scores, save_dir=save_dir_agg_set, dataset=dataset_name, metrics=metrics, rt=total_time_benchmark, run=-1)
            self.generate_plots(runs_plots_scores=scores, ticks=x_axis, metrics=metrics, subplot=True, y_size=y_p_size, save_dir=save_dir_agg_set, display=verb)

        print("\nThe results are saved in : ", save_dir, "\n")
        self.list_results = run_averaged
        self.aggregate_results = scores_list
