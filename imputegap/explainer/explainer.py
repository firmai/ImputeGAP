import math
import os

import numpy as np
import shap
import pycatch22
import toml
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor



from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation


class Explainer:

    def load_configuration(file_path="../env/default_explainer.toml"):
        """
        Load categories and features from a TOML file.

        :param file_path: The path to the TOML file.
        :return: Two dictionaries: categories and features.
        """
        # Load the TOML data from the file
        if not os.path.exists(file_path):
            file_path = file_path[4:]

        config_data = toml.load(file_path)

        # Extract categories and features from the TOML data
        categories = config_data.get('CATEGORIES', {})
        features = config_data.get('FEATURES', {})

        return categories, features


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
        data = np.array(data)

        if isinstance(data, np.ndarray):
            flat_data = data.flatten().tolist()
        else:
            flat_data = [float(item) for sublist in data for item in sublist]

        # If the data is a 2D list (similar to what's being read from the file),
        # then flatten it into a 1D list.
        if isinstance(flat_data[0], list):
            flat_data = [float(item) for sublist in flat_data for item in sublist]

        catch_out = pycatch22.catch22_all(flat_data, catch24=do_catch24)

        feature_names = catch_out['names']
        feature_values = catch_out['values']
        results = {}
        descriptions = []

        for feature_name, feature_value in zip(feature_names, feature_values):
            results[feature_name] = feature_value

            for category, features in features_categories.items():
                if feature_name in features:
                    category_value = category
                    break

            feature_description = features_list.get(feature_name)

            descriptions.append((feature_name, category_value, feature_description))

        print("\n%%%%% pycatch22 : features extracted successfully %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

        return results, descriptions

    def convert_results(tmp, file, algo, descriptions, features, categories, mean_features, rmse):
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
        :param rmse: RMSE score of the imputation
        :return: Perfect diplay for SHAP result
        """

        print("\n\n----------CONVERT : ", tmp)
        print("\n\n----------CONVERT : ", np.array(tmp).shape)

        result_display, display_details, result_shap = [], [], []
        for x, rate in enumerate(tmp):
            if math.isnan(rate) == False:
                rate = float(round(rate, 2))

            result_display.append(
                (x, algo, rate, descriptions[0][x], features[0][x], categories[0][x], mean_features[x], rmse))

        result_display = sorted(result_display, key=lambda tup: (tup[1], tup[2]), reverse=True)

        for (x, algo, rate, description, feature, categorie, mean_features, rmse) in result_display:
            print(x, " : ", algo, " with a score of ", rate, "  (", description, " / ", feature, " / ", categorie,
                  ")\n")
            result_shap.append([file, algo, rate, description, feature, categorie, mean_features, rmse])

        print("----------CONVERT : ", np.array(result_shap).shape)

        return result_shap

    def launch_shap_model(x_dataset, x_information, y_dataset, file, algorithm, splitter=10):
        """
        Launch the SHAP model for explaining the features of the dataset
        @author : Quentin Nater

        :param x_dataset:  Dataset of features extraction with descriptions
        :param x_information: Descriptions of all features group by categories
        :param y_dataset: Label RMSE of each series
        :param file: dataset used
        :param algorithm: algorithm used
        :param splitter: splitter from data training and testing
        :return: results of the explainer model
        """

        print("\n\n======= SHAP >> MODEL ======= shape set : ", np.array(x_information).shape,
              "======= ======= ======= ======= ======= ======= ======= ======= ======= ")

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
        print("\t SHAP_MODEL >> NATERQ x_train shape:", x_train.shape)
        print("\t SHAP_MODEL >> NATERQ y_train shape:", y_train.shape)
        print("\t SHAP_MODEL >> NATERQ x_test shape:", x_test.shape)
        print("\t SHAP_MODEL >> NATERQ y_test shape:", y_test.shape, "\n")
        print("\t SHAP_MODEL >> NATERQ x_features shape:", x_features.shape)
        print("\t SHAP_MODEL >> NATERQ x_categories shape:", x_categories.shape)
        print("\t SHAP_MODEL >> NATERQ x_descriptions shape:", x_descriptions.shape, "\n")
        print("\t SHAP_MODEL >> NATERQ FEATURES OK:", np.all(np.all(x_features == x_features[0, :], axis=1)))
        print("\t SHAP_MODEL >> NATERQ x_categories OK:", np.all(np.all(x_categories == x_categories[0, :], axis=1)))
        print("\t SHAP_MODEL >> NATERQ x_descriptions OK:",
              np.all(np.all(x_descriptions == x_descriptions[0, :], axis=1)), "\n\n")

        model = RandomForestRegressor()
        model.fit(x_train, y_train)

        # print("\t\t SHAP_MODEL >>  NATERQ model coefficients : \t", model.feature_importances_)

        exp = shap.KernelExplainer(model.predict, x_test)
        shval = exp.shap_values(x_test)
        shap_values = exp(x_train)

        print("\t\t SHAP_MODEL >>  NATERQ shval selected : ", np.array(shval).shape,
              "************************************")
        print("\t\t SHAP_MODEL >>  NATERQ shval selected : \t", *shval)

        optimal_display = []
        for desc, group in zip(x_descriptions[0], x_categories[0]):
            optimal_display.append(desc + " (" + group + ")")

        series_names = []
        for names in range(0, np.array(x_test).shape[0]):
            series_names.append("Series " + str(names + np.array(x_train).shape[0]))

        shap.summary_plot(shval, x_test, plot_size=(25, 10), feature_names=optimal_display)
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_shap_plot.png"
        plt.title("SHAP Details Results")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(shval).T, np.array(x_test).T, feature_names=series_names)
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_shap_reverse_plot.png"
        plt.title("SHAP Features by Series")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

        shap.plots.waterfall(shap_values[0])
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_DTL_Waterfall.png"
        plt.title("SHAP Waterfall Results")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

        shap.plots.beeswarm(shap_values)
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_DTL_Beeswarm.png"
        plt.title("SHAP Beeswarm Results")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

        print("\n\n\t\t\tSHAP_BUILD_____________________________________________________________________")
        total_weights_for_all_algorithms = []

        t_shval = np.array(shval).T
        t_Xtest = np.array(x_test).T

        aggregation_features, aggregation_test = [], []

        print("\t\t\tSHAP_BUILD >>  NATERQ t_shval shape : ", np.array(t_shval).shape,
              "************************************")
        print("\t\t\tSHAP_BUILD >>  NATERQ t_Xtest shape : ", np.array(t_Xtest).shape)

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

        print("\n\t\t\tSHAP_BUILD geometry:", geometry.shape)
        print("\n\t\t\tSHAP_BUILD geometryT:", geometryT.shape)
        print("\n\t\t\tSHAP_BUILD transformation:", transformation.shape)
        print("\n\t\t\tSHAP_BUILD transformationT:", transformationT.shape)
        print("\n\t\t\tSHAP_BUILD correlation:", correlation.shape)
        print("\n\t\t\tSHAP_BUILD correlationT:", correlationT.shape)
        print("\n\t\t\tSHAP_BUILD trend':", trend.shape)
        print("\n\t\t\tSHAP_BUILD trendT:", trendT.shape)
        print("\n\t\t\tSHAP_BUILD mean_features:", mean_features.shape)

        shap.summary_plot(np.array(geometry).T, np.array(geometryT).T, plot_size=(20, 10), feature_names=geometryDesc)
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_shap_geometry_plot.png"
        plt.title("SHAP details of geometry")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(transformation).T, np.array(transformationT).T, plot_size=(20, 10),
                          feature_names=transformationDesc)
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_shap_transformation_plot.png"
        plt.title("SHAP details of transformation")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(correlation).T, np.array(correlationT).T, plot_size=(20, 10),
                          feature_names=correlationDesc)
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_shap_correlation_plot.png"
        plt.title("SHAP details of correlation")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(trend).T, np.array(trendT).T, plot_size=(20, 8), feature_names=trendDesc)
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_shap_trend_plot.png"
        plt.title("SHAP details of Trend")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

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
                          feature_names=['Geometry', 'Correlation', 'Transformation', 'Trend'])
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_shap_aggregate_plot.png"
        plt.title("SHAP Aggregation Results")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

        shap.summary_plot(np.array(aggregation_features).T, np.array(aggregation_test).T, feature_names=series_names)
        alpha = "parameterizer_frontend/src/assets_naterq/" + file + "_" + algorithm + "_shap_aggregate_reverse_plot.png"
        plt.title("SHAP Aggregation Features by Series")
        plt.savefig(alpha)
        plt.close()
        print("\t\t\t SHAP_MODEL >>  GRAPH has benn computed : ", alpha)

        # Aggregate shapely values per element of X_test
        total_weights = [np.abs(shval.T[i]).mean(0) for i in range(len(shval[0]))]

        # Convert to percentages
        total_sum = np.sum(total_weights)
        total_weights_percent = [(weight / total_sum * 100) for weight in total_weights]

        total_weights_for_all_algorithms = np.append(total_weights_for_all_algorithms, total_weights_percent)

        results_shap = Explainer.convert_results(total_weights_for_all_algorithms, file, algorithm, x_descriptions, x_features, x_categories, mean_features, y_dataset.tolist())

        return results_shap

    def shap_explainer(ground_truth, algorithm="cdrec", params=None, contamination="mcar", missing_rate=0.4, block_size=10, protection=0.1, use_seed=True, seed=42, limitation=15, splitter=0, nbr_values=800):
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
        :param nbr_values: limitation of the number of values for each series | default 800

        :return: ground_truth_matrixes, obfuscated_matrixes, output_metrics, input_params, shap_values
        """

        print("°°°°°SHAP >> NATERQ params : missing_values (", missing_rate, ") \n",
        "for a contamination (", contamination, "), \n",
        "limited to (", limitation, ") with splitter (", splitter, ") \n")

        print("°°°°°SHAP >> NATERQ params : algo (", algorithm, ") / params (", params, ")\n")

        if limitation > ground_truth.shape[0]:
            limitation = int(ground_truth.shape[0]*0.75)

        if splitter == 0 or splitter >= limitation-1 :
            splitter = int(limitation*0.60)

        print("\n\t\t\t\t\t°°°°°SHAP >> Splitter and Limitation after verification ", ground_truth.shape, "\n\t\t",
                  " >> (limitation = ", limitation, " and splitter = ", splitter, ")...\n")

        ground_truth_matrixes, obfuscated_matrixes = [], []
        output_metrics, output_rmse, input_params, input_params_full = [], [], [], []

        categories, features = Explainer.load_configuration()

        for current_series in range(0, limitation):

            if contamination == "mcar":
                obfuscated_matrix = Contamination.scenario_mcar(ts=ground_truth, series_impacted=current_series, missing_rate=missing_rate, block_size=block_size, protection=protection, use_seed=use_seed, seed=seed, explainer=True)
            else:
                print("Contamination proposed not found : ", contamination, " >> BREAK")
                return None

            ground_truth_matrixes.append(ground_truth)
            obfuscated_matrixes.append(obfuscated_matrix)

            if algorithm == "cdrec":
                _, imputation_results = Imputation.MR.cdrec(ground_truth, obfuscated_matrix, params)
            elif algorithm == "stmvl":
                _, imputation_results = Imputation.Pattern.stmvl_imputation(ground_truth, obfuscated_matrix, params)
            elif algorithm == "iim":
                _, imputation_results = Imputation.Regression.iim_imputation(ground_truth, obfuscated_matrix, params)
            elif algorithm == "mrnn":
                _, imputation_results = Imputation.ML.mrnn_imputation(ground_truth, obfuscated_matrix, params)

            output_metrics.append(imputation_results)
            output_rmse.append(imputation_results["RMSE"])

            catch_fct, descriptions = Explainer.extract_features(np.array(obfuscated_matrix), categories, features, False)

            extracted_features = np.array(list(catch_fct.values()))

            input_params.append(extracted_features)
            input_params_full.append(descriptions)

            print(
                "°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
            print("\t\t°°°°°SHAP >> NATERQ Current series contamination : ", current_series, " °°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
            print("\t\t°°°°°SHAP >> NATERQ SHAPE TEST : ", np.array(ground_truth).shape)
            print("\t\t°°°°°SHAP >> NATERQ SHAPE TEST : ", np.array(obfuscated_matrix).shape)
            print("\t\t°°°°°SHAP >> NATERQ Current series ", current_series, " contamination done")
            print("\t\t°°°°°SHAP >> NATERQ Current series ", current_series, " imputation done")
            print(
                "°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        for read in output_metrics:
            print("°°°°°SHAP >> NATERQ RESULTS_: Metrics RMSE : ", read["RMSE"])

        shap_values = Explainer.launch_shap_model(input_params, input_params_full, output_rmse, ground_truth, algorithm, splitter)

        print(
            "°°°°°SHAP >> NATERQ SHAP COMPUTED AND ENDED SUCCESSFULLY ! °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°\n\n\n")

        return ground_truth_matrixes, obfuscated_matrixes, output_metrics, input_params, shap_values


