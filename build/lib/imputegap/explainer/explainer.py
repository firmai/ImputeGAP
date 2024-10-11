import math
import os

import numpy as np
import shap
import pycatch22
import toml
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries


class Explainer:

    def load_configuration(file_path="../env/default_explainer.toml"):
        """
        Load categories and features from a TOML file.

        :param file_path: The path to the TOML file.
        :return: Two dictionaries: categories and features.
        """
        if not os.path.exists(file_path):
            file_path = file_path[1:]

        config_data = toml.load(file_path)

        # Extract categories and features from the TOML data
        categories = config_data.get('CATEGORIES', {})
        features = config_data.get('FEATURES', {})

        return categories, features

    def save_assets(file_path="./assets/shap/"):
        """
        Load path to save the assets on GitHub and local
        :param file_path: The path to the TOML file.
        :return: Two dictionaries: categories and features.
        """
        if not os.path.exists(file_path):
            file_path = "./imputegap" + file_path[1:]

        return file_path

    def extract_features(data, features_categories, features_list, do_catch24=True):
        """
        Extract features from time series data using pycatch22.
        @author : Quentin Nater

        :param data : time series dataset to extract features
        :param features_categories : way to category the features
        :param features_list : list of all features expected
        :param do_catch24 : Flag to compute the mean and standard deviation. Defaults to True.

        :return : results, descriptions : dictionary of feature values by names, and array of their descriptions.
        """
        data = [[0 if num is None else num for num in sublist] for sublist in data]
        data = [[0 if num is None or (isinstance(num, (float, np.float32, np.float64)) and np.isnan(num)) else num for num in sublist] for sublist in data]

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
        Convert the SHAP brute result to a refined one to display in the front end
        @author : Quentin Nater

        :param shap_values: Values and results of the SHAP analytics
        :param shap_details: Input and Ouput data of the Regression
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
            print(f"\tFeature : {x:<5} {algo:<10} with a score of {rate:<10} {category:<18} {description:<75} {feature}\n")

    def convert_results(tmp, file, algo, descriptions, features, categories, mean_features, to_save):
        """
        Convert the SHAP brute result to a refined one to display in the front end
        @author : Quentin Nater

        :param tmp: Current results
        :param file: Dataset used
        :param algo: Algorithm used
        :param descriptions: Description of each feature
        :param features: Raw name of each feature
        :param categories: Category of each feature
        :param mean_features: Mean values of each feature
        :param to_save : path to save results
        :return: Perfect display for SHAP result
        """
        result_display, result_shap = [], []
        for x, rate in enumerate(tmp):
            if not math.isnan(rate):
                rate = float(round(rate, 2))

            result_display.append((x, algo, rate, descriptions[0][x], features[0][x], categories[0][x], mean_features[x]))

        result_display = sorted(result_display, key=lambda tup: (tup[1], tup[2]), reverse=True)

        for tup in result_display:
            print(tup[2], end=",")

        with open(to_save + "_results.txt", 'w') as file_output:
            for (x, algo, rate, description, feature, category, mean_features) in result_display:
                file_output.write(f"Feature : {x:<5} {algo:<10} with a score of {rate:<10} {category:<18} {description:<65} {feature}\n")
                result_shap.append([file, algo, rate, description, feature, category, mean_features])

        return result_shap

    def launch_shap_model(x_dataset, x_information, y_dataset, file, algorithm, splitter=10, display=False):
        """
        Launch the SHAP model for explaining the features of the dataset
        @author : Quentin Nater

        :param x_dataset:  Dataset of features extraction with descriptions
        :param x_information: Descriptions of all features group by categories
        :param y_dataset: Label RMSE of each series
        :param file: dataset used
        :param algorithm: algorithm used
        :param splitter: splitter from data training and testing
        :param display: display or not plots
        :return: results of the explainer model
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
        print("\t SHAP_MODEL >> features shape:", x_features.shape)
        print("\t SHAP_MODEL >> categories shape:", x_categories.shape)
        print("\t SHAP_MODEL >> descriptions shape:", x_descriptions.shape, "\n")
        print("\t SHAP_MODEL >> features OK:", np.all(np.all(x_features == x_features[0, :], axis=1)))
        print("\t SHAP_MODEL >> categories OK:", np.all(np.all(x_categories == x_categories[0, :], axis=1)))
        print("\t SHAP_MODEL >> descriptions OK:", np.all(np.all(x_descriptions==x_descriptions[0, :], axis=1)), "\n\n")

        model = RandomForestRegressor()
        model.fit(x_train, y_train)

        exp = shap.KernelExplainer(model.predict, x_test)
        shval = exp.shap_values(x_test)
        shap_values = exp(x_train)

        #print("\t\tSHAP VALUES : ", np.array(shval).shape, " with : \n\t", *shval)

        optimal_display = []
        for desc, group in zip(x_descriptions[0], x_categories[0]):
            optimal_display.append(desc + " (" + group + ")")

        series_names = []
        for names in range(0, np.array(x_test).shape[0]):
            series_names.append("Series " + str(names + np.array(x_train).shape[0]))

        shap.summary_plot(shval, x_test, plot_size=(25, 10), feature_names=optimal_display, show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_shap_plot.png")
        plt.title("SHAP Details Results")
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

        print("\t\tSHAP Families details : \n")
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

        results_shap = Explainer.convert_results(total_weights_for_all_algorithms, file, algorithm, x_descriptions, x_features, x_categories, mean_features, to_save=path_file + file + "_" + algorithm)

        return results_shap

    def shap_explainer(raw_data, algorithm="cdrec", params=None, contamination="mcar", missing_rate=0.4,
                       block_size=10, protection=0.1, use_seed=True, seed=42, limitation=15, splitter=0,
                       file_name="ts", display=False):
        """
        Handle parameters and set the variables to launch a model SHAP
        @author : Quentin Nater

        :param dataset: imputegap dataset used for timeseries
        :param algorithm: [OPTIONAL] algorithm used for imputation ("cdrec", "stvml", "iim", "mrnn") | default : cdrec
        :param params: [OPTIONAL] parameters of algorithms
        :param contamination: scenario used to contaminate the series | default mcar
        :param missing_rate: percentage of missing values by series  | default 0.2
        :param block_size: size of the block to remove at each random position selected  | default 10
        :param protection: size in the beginning of the time series where contamination is not proceeded  | default 0.1
        :param use_seed: use a seed to reproduce the test | default true
        :param seed: value of the seed | default 42
        :param limitation: limitation of series for the model | default 15
        :param splitter: limitation of training series for the model | default 3/4 of limitation
        :param display: display or not the plots | default False

        :return: ground_truth_matrixes, obfuscated_matrixes, output_metrics, input_params, shap_values
        """

        if limitation > raw_data.shape[0]:
            limitation = int(raw_data.shape[0] * 0.75)

        if splitter == 0 or splitter >= limitation - 1:
            splitter = int(limitation * 0.60)

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
            print("Generation ", current_series, "___________________________________________________________________")
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
                algo = Imputation.MD.CDRec(obfuscated_matrix)
            elif algorithm == "stmvl":
                algo = Imputation.Pattern.STMVL(obfuscated_matrix)
            elif algorithm == "iim":
                algo = Imputation.Regression.IIM(obfuscated_matrix)
            elif algorithm == "mrnn":
                algo = Imputation.ML.MRNN(obfuscated_matrix)

            algo.impute(user_defined=True, params=params)
            algo.score(raw_data)
            imputation_results = algo.metrics

            output_metrics.append(imputation_results)
            output_rmse.append(imputation_results["RMSE"])

            catch_fct, descriptions = Explainer.extract_features(np.array(obfuscated_matrix), categories, features, False)

            extracted_features = np.array(list(catch_fct.values()))

            input_params.append(extracted_features)
            input_params_full.append(descriptions)

        shap_details = []
        for input, output in zip(input_params, output_metrics):
            shap_details.append((input, output["RMSE"]))

        shap_values = Explainer.launch_shap_model(input_params, input_params_full, output_rmse, file_name, algorithm, splitter, display)

        print("\n\n\nSHAP Explainer succeeded without fail, please find the results in : ./assets/shap/*\n\n\n")

        return shap_values, shap_details
