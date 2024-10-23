import math
import os
import time
import importlib.resources

import numpy as np
import shap
import pycatch22
import toml
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries


class Explainer:
    """
    A class to manage SHAP-based model explanations and feature extraction for time series datasets.

    Methods
    -------
    load_configuration(file_path=None)
        Load categories and features from a TOML file.

    save_assets(file_path="./assets/shap/")
        Load path to save SHAP-related assets.

    extract_features(data, features_categories, features_list, do_catch24=True)
        Extract features from time series data using pycatch22.

    print(shap_values, shap_details=None)
        Print SHAP values and details for display.

    convert_results(tmp, file, algo, descriptions, features, categories, mean_features, to_save)
        Convert SHAP raw results into a refined format for display.

    launch_shap_model(x_dataset, x_information, y_dataset, file, algorithm, splitter=10, display=False, verbose=False)
        Launch the SHAP model to explain the dataset features.

    shap_explainer(raw_data, algorithm="cdrec", params=None, contamination="mcar", missing_rate=0.4,
                   block_size=10, protection=0.1, use_seed=True, seed=42, limitation=15, splitter=0,
                   file_name="ts", display=False, verbose=False)
        Handle parameters and set variables to launch the SHAP model.

    """

    def load_configuration(file_path=None):
        """
        Load categories and features from a TOML file.

        Parameters
        ----------
        file_path : str, optional
            The path to the TOML file (default is None). If None, it loads the default configuration file.

        Returns
        -------
        tuple
            A tuple containing two dictionaries: categories and features.
        """

        if file_path is None:
            path = importlib.resources.files('imputegap.env').joinpath("./default_explainer.toml")
        else:
            if not os.path.exists(file_path):
                file_path = file_path[1:]

        config_data = toml.load(path)

        # Extract categories and features from the TOML data
        categories = config_data.get('CATEGORIES', {})
        features = config_data.get('FEATURES', {})

        return categories, features

    def save_assets(file_path="./assets/shap/"):
        """
        Load path to save SHAP-related assets on GitHub and local.

        Parameters
        ----------
        file_path : str, optional
            The path to save the SHAP assets (default is './assets/shap/').

        Returns
        -------
        str
            The file path where assets are saved.
        """

        if not os.path.exists(file_path):
            file_path = "./imputegap" + file_path[1:]

        return file_path

    def extract_features(data, features_categories, features_list, do_catch24=True):
        """
        Extract features from time series data using pycatch22.

        Parameters
        ----------
        data : numpy.ndarray
            Time series dataset for feature extraction.
        features_categories : dict
            Dictionary that maps feature names to categories.
        features_list : dict
            Dictionary of all features expected.
        do_catch24 : bool, optional
            Flag to compute the mean and standard deviation for Catch24 (default is True).

        Returns
        -------
        tuple
            A tuple containing:
            - results (dict): A dictionary of feature values by feature names.
            - descriptions (list): A list of tuples containing feature names, categories, and descriptions.
        """

        data = [[0 if num is None else num for num in sublist] for sublist in data]
        data = [
            [0 if num is None or (isinstance(num, (float, np.float32, np.float64)) and np.isnan(num)) else num for num
             in sublist] for sublist in data]

        data = np.array(data)

        if isinstance(data, np.ndarray):
            flat_data = data.flatten().tolist()
        else:
            flat_data = [float(item) for sublist in data for item in sublist]

        if isinstance(flat_data[0], list):
            flat_data = [float(item) for sublist in flat_data for item in sublist]

        catch_out = pycatch22.catch22_all(flat_data, catch24=do_catch24)

        feature_names = catch_out['names']
        feature_values = catch_out['values']
        results, descriptions = {}, []

        if any(isinstance(value, (float, np.float32, np.float64)) and np.isnan(value) for value in feature_values):
            raise ValueError("Error: NaN value detected in feature_values")

        for feature_name, feature_value in zip(feature_names, feature_values):
            results[feature_name] = feature_value

            for category, features in features_categories.items():
                if feature_name in features:
                    category_value = category
                    break

            feature_description = features_list.get(feature_name)

            descriptions.append((feature_name, category_value, feature_description))

        print("pycatch22 : features extracted successfully_______________________________________\n\n")

        return results, descriptions

    def print(shap_values, shap_details=None):
        """
        Convert SHAP raw results to a refined format for display.

        Parameters
        ----------
        shap_values : list
            The SHAP values and results of the SHAP analysis.
        shap_details : list, optional
            Input and output data of the regression, if available (default is None).

        Returns
        -------
        None
        """

        if shap_details is not None:
            print("\n\nx_data (with", len(shap_details), "elements) : ")
            for i, (input, _) in enumerate(shap_details):
                print("\tFEATURES VALUES", i, "(", len(input), ") : ", *input)

            print("\ny_data (with", len(shap_details), "elements) : ")

            for i, (_, output) in enumerate(shap_details):
                print(f"\tRMSE SERIES {i:<5} : {output:<15}")

        print("\n\nSHAP Results details : ")
        for (x, algo, rate, description, feature, category, mean_features) in shap_values:
            print(
                f"\tFeature : {x:<5} {algo:<10} with a score of {rate:<10} {category:<18} {description:<75} {feature}\n")

    def convert_results(tmp, file, algo, descriptions, features, categories, mean_features, to_save):
        """
        Convert SHAP raw results to a refined format for display.

        Parameters
        ----------
        tmp : list
            Current SHAP results.
        file : str
            Dataset used.
        algo : str
            Algorithm used for imputation.
        descriptions : list
            Descriptions of each feature.
        features : list
            Raw names of each feature.
        categories : list
            Categories of each feature.
        mean_features : list
            Mean values of each feature.
        to_save : str
            Path to save results.

        Returns
        -------
        list
            A list of processed SHAP results.
        """

        result_display, result_shap = [], []
        for x, rate in enumerate(tmp):
            if not math.isnan(rate):
                rate = float(round(rate, 2))

            result_display.append(
                (x, algo, rate, descriptions[0][x], features[0][x], categories[0][x], mean_features[x]))

        result_display = sorted(result_display, key=lambda tup: (tup[1], tup[2]), reverse=True)

        with open(to_save + "_results.txt", 'w') as file_output:
            for (x, algo, rate, description, feature, category, mean_features) in result_display:
                file_output.write(
                    f"Feature : {x:<5} {algo:<10} with a score of {rate:<10} {category:<18} {description:<65} {feature}\n")
                result_shap.append([file, algo, rate, description, feature, category, mean_features])

        return result_shap

    def launch_shap_model(x_dataset, x_information, y_dataset, file, algorithm, splitter=10, display=False,
                          verbose=False):
        """
        Launch the SHAP model for explaining the features of the dataset.

        Parameters
        ----------
        x_dataset : numpy.ndarray
            Dataset of feature extraction with descriptions.
        x_information : list
            Descriptions of all features grouped by categories.
        y_dataset : numpy.ndarray
            RMSE labels of each series.
        file : str
            Dataset used for SHAP analysis.
        algorithm : str
            Algorithm used for imputation (e.g., 'cdrec', 'stmvl', 'iim', 'mrnn').
        splitter : int, optional
            Split ratio for data training and testing (default is 10).
        display : bool, optional
            Whether to display the SHAP plots (default is False).
        verbose : bool, optional
            Whether to print detailed output (default is False).

        Returns
        -------
        list
            Results of the SHAP explainer model.
        """

        print("\n\nInitilization of the SHAP model with ", np.array(x_information).shape)

        path_file = Explainer.save_assets()

        x_features, x_categories, x_descriptions = [], [], []
        x_fs, x_cs, x_ds = [], [], []

        for current_time_series in x_information:
            x_fs.clear()
            x_cs.clear()
            x_ds.clear()
            for feature_name, category_value, feature_description in current_time_series:
                x_fs.append(feature_name)
                x_cs.append(category_value)
                x_ds.append(feature_description)
            x_features.append(x_fs)
            x_categories.append(x_cs)
            x_descriptions.append(x_ds)

        x_dataset = np.array(x_dataset)
        y_dataset = np.array(y_dataset)

        x_features = np.array(x_features)
        x_categories = np.array(x_categories)
        x_descriptions = np.array(x_descriptions)

        # Split the data
        x_train, x_test = x_dataset[:splitter], x_dataset[splitter:]
        y_train, y_test = y_dataset[:splitter], y_dataset[splitter:]

        # Print shapes to verify
        print("\t SHAP_MODEL >> x_train shape:", x_train.shape)
        print("\t SHAP_MODEL >> y_train shape:", y_train.shape)
        print("\t SHAP_MODEL >> x_test shape:", x_test.shape)
        print("\t SHAP_MODEL >> y_test shape:", y_test.shape, "\n")
        if verbose:
            print("\t SHAP_MODEL >> features shape:", x_features.shape)
            print("\t SHAP_MODEL >> categories shape:", x_categories.shape)
            print("\t SHAP_MODEL >> descriptions shape:", x_descriptions.shape, "\n")
            print("\t SHAP_MODEL >> features OK:", np.all(np.all(x_features == x_features[0, :], axis=1)))
            print("\t SHAP_MODEL >> categories OK:", np.all(np.all(x_categories == x_categories[0, :], axis=1)))
            print("\t SHAP_MODEL >> descriptions OK:", np.all(np.all(x_descriptions == x_descriptions[0, :], axis=1)),
                  "\n\n")

        model = RandomForestRegressor()
        model.fit(x_train, y_train)

        exp = shap.KernelExplainer(model.predict, x_test)
        shval = exp.shap_values(x_test)
        shap_values = exp(x_train)

        optimal_display = []
        for desc, group in zip(x_descriptions[0], x_categories[0]):
            optimal_display.append(desc + " (" + group + ")")

        series_names = []
        for names in range(0, np.array(x_test).shape[0]):
            series_names.append("Series " + str(names + np.array(x_train).shape[0]))

        shap.summary_plot(shval, x_test, plot_size=(25, 10), feature_names=optimal_display, show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_shap_plot.png")
        plt.title("SHAP Details Results")
        os.makedirs(path_file, exist_ok=True)
        plt.savefig(alpha)
        plt.close()
        print("\n\n\t\t\tGRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(shval).T, np.array(x_test).T, feature_names=series_names, show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_shap_reverse_plot.png")
        plt.title("SHAP Features by Series")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\tGRAPH has benn computed : ", alpha)

        shap.plots.waterfall(shap_values[0], show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_DTL_Waterfall.png")
        plt.title("SHAP Waterfall Results")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\tGRAPH has benn computed : ", alpha)

        shap.plots.beeswarm(shap_values, show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_DTL_Beeswarm.png")
        plt.title("SHAP Beeswarm Results")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\tGRAPH has benn computed : ", alpha)

        total_weights_for_all_algorithms = []

        t_shval = np.array(shval).T
        t_Xtest = np.array(x_test).T

        aggregation_features, aggregation_test = [], []

        geometry, correlation, transformation, trend = [], [], [], []
        geometryDesc, correlationDesc, transformationDesc, trendDesc = [], [], [], []

        for index, feat in enumerate(t_shval):
            if x_categories[0][index] == "Geometry":
                geometry.append(feat)
                geometryDesc.append(x_descriptions[0][index])
            elif x_categories[0][index] == "Correlation":
                correlation.append(feat)
                correlationDesc.append(x_descriptions[0][index])
            elif x_categories[0][index] == "Transformation":
                transformation.append(feat)
                transformationDesc.append(x_descriptions[0][index])
            elif x_categories[0][index] == "Trend":
                trend.append(feat)
                trendDesc.append(x_descriptions[0][index])

        geometryT, correlationT, transformationT, trendT = [], [], [], []
        for index, feat in enumerate(t_Xtest):
            if x_categories[0][index] == "Geometry":
                geometryT.append(feat)
            elif x_categories[0][index] == "Correlation":
                correlationT.append(feat)
            elif x_categories[0][index] == "Transformation":
                transformationT.append(feat)
            elif x_categories[0][index] == "Trend":
                trendT.append(feat)

        mean_features = []
        for feat in t_Xtest:
            mean_features.append(np.mean(feat, axis=0))

        geometry = np.array(geometry)
        correlation = np.array(correlation)
        transformation = np.array(transformation)
        trend = np.array(trend)
        geometryT = np.array(geometryT)
        correlationT = np.array(correlationT)
        transformationT = np.array(transformationT)
        trendT = np.array(trendT)
        mean_features = np.array(mean_features)

        shap.summary_plot(np.array(geometry).T, np.array(geometryT).T, plot_size=(20, 10), feature_names=geometryDesc,
                          show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_shap_geometry_plot.png")
        plt.title("SHAP details of geometry")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\tGRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(transformation).T, np.array(transformationT).T, plot_size=(20, 10),
                          feature_names=transformationDesc, show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_shap_transformation_plot.png")
        plt.title("SHAP details of transformation")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\tGRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(correlation).T, np.array(correlationT).T, plot_size=(20, 10),
                          feature_names=correlationDesc, show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_shap_correlation_plot.png")
        plt.title("SHAP details of correlation")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\tGRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(trend).T, np.array(trendT).T, plot_size=(20, 8), feature_names=trendDesc,
                          show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_shap_trend_plot.png")
        plt.title("SHAP details of Trend")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\tGRAPH has benn computed : ", alpha)

        aggregation_features.append(np.mean(geometry, axis=0))
        aggregation_features.append(np.mean(correlation, axis=0))
        aggregation_features.append(np.mean(transformation, axis=0))
        aggregation_features.append(np.mean(trend, axis=0))

        aggregation_test.append(np.mean(geometryT, axis=0))
        aggregation_test.append(np.mean(correlationT, axis=0))
        aggregation_test.append(np.mean(transformationT, axis=0))
        aggregation_test.append(np.mean(trendT, axis=0))

        aggregation_features = np.array(aggregation_features).T
        aggregation_test = np.array(aggregation_test).T

        shap.summary_plot(aggregation_features, aggregation_test,
                          feature_names=['Geometry', 'Correlation', 'Transformation', 'Trend'], show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_shap_aggregate_plot.png")
        plt.title("SHAP Aggregation Results")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.savefig(alpha)
        plt.close()
        print("\t\t\tGRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(aggregation_features).T, np.array(aggregation_test).T, feature_names=series_names,
                          show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_shap_aggregate_reverse_plot.png")
        plt.title("SHAP Aggregation Features by Series")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\tGRAPH has benn computed : ", alpha, "\n\n")

        if verbose:
            print("\t\tSHAP Families details :")
            print("\t\t\tgeometry:", geometry.shape)
            print("\t\t\ttransformation:", transformation.shape)
            print("\t\t\tcorrelation:", correlation.shape)
            print("\t\t\ttrend':", trend.shape)
            print("\t\t\tmean_features:", mean_features.shape, "\n\n")

        # Aggregate shapely values per element of X_test
        total_weights = [np.abs(shval.T[i]).mean(0) for i in range(len(shval[0]))]

        # Convert to percentages
        total_sum = np.sum(total_weights)
        total_weights_percent = [(weight / total_sum * 100) for weight in total_weights]

        total_weights_for_all_algorithms = np.append(total_weights_for_all_algorithms, total_weights_percent)

        results_shap = Explainer.convert_results(total_weights_for_all_algorithms, file, algorithm, x_descriptions,
                                                 x_features, x_categories, mean_features,
                                                 to_save=path_file + file + "_" + algorithm)

        return results_shap

    def shap_explainer(raw_data, algorithm="cdrec", params=None, contamination="mcar", missing_rate=0.4,
                       block_size=10, protection=0.1, use_seed=True, seed=42, limitation=15, splitter=0,
                       file_name="ts", display=False, verbose=False):
        """
        Handle parameters and set variables to launch the SHAP model.

        Parameters
        ----------
        raw_data : numpy.ndarray
            The original time series dataset.
        algorithm : str, optional
            The algorithm used for imputation (default is 'cdrec'). Valid values: 'cdrec', 'stmvl', 'iim', 'mrnn'.
        params : dict, optional
            Parameters for the algorithm.
        contamination : str, optional
            Contamination scenario to apply (default is 'mcar').
        missing_rate : float, optional
            Percentage of missing values per series (default is 0.4).
        block_size : int, optional
            Size of the block to remove at each random position selected (default is 10).
        protection : float, optional
            Size of the uncontaminated section at the beginning of the time series (default is 0.1).
        use_seed : bool, optional
            Whether to use a seed for reproducibility (default is True).
        seed : int, optional
            Seed value for reproducibility (default is 42).
        limitation : int, optional
            Limitation on the number of series for the model (default is 15).
        splitter : int, optional
            Limitation on the training series for the model (default is 0).
        file_name : str, optional
            Name of the dataset file (default is 'ts').
        display : bool, optional
            Whether to display the SHAP plots (default is False).
        verbose : bool, optional
            Whether to print detailed output (default is False).

        Returns
        -------
        tuple
            A tuple containing:

            - shap_values : list
                SHAP values for each series.
            - shap_details : list
                Detailed SHAP analysis results.

        Notes
        -----
        The contamination is applied to each time series using the specified method. The SHAP model is then used
        to generate explanations for the imputation results, which are logged in a local directory.
        """

        start_time = time.time()  # Record start time

        if limitation > raw_data.shape[0]:
            limitation = int(raw_data.shape[0] * 0.75)

        if splitter == 0 or splitter >= limitation - 1:
            splitter = int(limitation * 0.60)

        if verbose:
            print("SHAP Explainer has been called\n\t",
                  "missing_values (", missing_rate * 100, "%)\n\t",
                  "for a contamination (", contamination, "), \n\t",
                  "imputated by (", algorithm, ") with params (", params, ")\n\t",
                  "with limitation and splitter after verification of (", limitation, ") and (", splitter, ") for ",
                  raw_data.shape, "...\n\n\tGeneration of the dataset with the time series...")

        ground_truth_matrices, obfuscated_matrices = [], []
        output_metrics, output_rmse, input_params, input_params_full = [], [], [], []

        categories, features = Explainer.load_configuration()

        for current_series in range(0, limitation):

            print("Generation ", current_series, "/", limitation, "(", int((current_series / limitation) * 100),
                  "%)________________________________________________________")
            print("\tContamination ", current_series, "...")

            if contamination == "mcar":
                obfuscated_matrix = TimeSeries().Contaminate.mcar(ts=raw_data, series_impacted=current_series,
                                                                  missing_rate=missing_rate, block_size=block_size,
                                                                  protection=protection, use_seed=use_seed, seed=seed,
                                                                  explainer=True)
            else:
                print("Contamination proposed not found : ", contamination, " >> BREAK")
                return None

            ground_truth_matrices.append(raw_data)
            obfuscated_matrices.append(obfuscated_matrix)

            print("\tImputation ", current_series, "...")
            if algorithm == "cdrec":
                algo = Imputation.MatrixCompletion.CDRec(obfuscated_matrix)
            elif algorithm == "stmvl":
                algo = Imputation.PatternSearch.STMVL(obfuscated_matrix)
            elif algorithm == "iim":
                algo = Imputation.Statistics.IIM(obfuscated_matrix)
            elif algorithm == "mrnn":
                algo = Imputation.DeepLearning.MRNN(obfuscated_matrix)

            algo.logs = False
            algo.impute(user_defined=True, params=params)
            algo.score(raw_data)
            imputation_results = algo.metrics

            output_metrics.append(imputation_results)
            output_rmse.append(imputation_results["RMSE"])

            catch_fct, descriptions = Explainer.extract_features(np.array(obfuscated_matrix), categories, features,
                                                                 False)

            extracted_features = np.array(list(catch_fct.values()))

            input_params.append(extracted_features)
            input_params_full.append(descriptions)

        shap_details = []
        for input, output in zip(input_params, output_metrics):
            shap_details.append((input, output["RMSE"]))

        shap_values = Explainer.launch_shap_model(input_params, input_params_full, output_rmse, file_name, algorithm,
                                                  splitter, display, verbose)

        print("\n\nSHAP Explainer succeeded without fail, please find the results in : ./assets/shap/*\n")

        end_time = time.time()
        print(f"\n\t\t> logs, shap explainer - Execution Time: {(end_time - start_time):.4f} seconds\n\n\n")

        return shap_values, shap_details
