=========
Tutorials
=========

.. _loading-preprocessing:

Loading and Preprocessing
-------------------------

The data management module allows loading any time series datasets in text format: *(values, series)*.

**Example Loading**

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("eeg-alcohol"), max_series=5, max_values=15)
    ts_1.normalize(normalizer="z_score")

    # [OPTIONAL] you can plot your raw data / print the information
    ts_1.plot(input_data=ts_1.data, max_series=10, max_values=100, save_path="./imputegap/assets")
    ts_1.print(limit_series=10)

.. _contamination:

Contamination
-------------

ImputeGAP allows adding missing data patterns such as `MCAR`, `BLACKOUT`, `GAUSSIAN`, etc.

**Example Contamination**

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("eeg-alcohol"))
    ts_1.normalize(normalizer="min_max")

    # 3. contamination of the data with MCAR pattern
    ts_mask = ts_1.Contamination.mcar(ts_1.data, rate_dataset=0.2, rate_series=0.2, seed=True)

    # [OPTIONAL] you can plot your raw data / print the contamination
    ts_1.print(limit_timestamps=12, limit_series=7)
    ts_1.plot(ts_1.data, ts_mask, max_series=9, subplot=True, save_path="./imputegap/assets")


.. _imputation:

Imputation
----------

ImputeGAP provides multiple imputation algorithms: Matrix Completion, Deep Learning, and Statistical Methods.

**Example Imputation**

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("eeg-alcohol"))
    ts_1.normalize(normalizer="min_max")

    # 3. contamination of the data
    ts_mask = ts_1.Contamination.mcar(ts_1.data)

    # [OPTIONAL] save your results in a new Time Series object
    ts_2 = TimeSeries().import_matrix(ts_mask)

    # 4. imputation of the contaminated data
    imputer = Imputation.MatrixCompletion.CDRec(ts_2.data)

    # imputation with default values
    imputer.impute()
    # OR imputation with user defined values
    # >>> cdrec.impute(params={"rank": 5, "epsilon": 0.01, "iterations": 100})

    # [OPTIONAL] save your results in a new Time Series object
    ts_3 = TimeSeries().import_matrix(imputer.recov_data)

    # 5. score the imputation with the raw_data
    imputer.score(ts_1.data, ts_3.data)

    # 6. display the results
    ts_3.print_results(imputer.metrics, algorithm=imputer.algorithm)
    ts_3.plot(input_data=ts_1.data, incomp_data=ts_2.data, recov_data=ts_3.data, max_series=9, subplot=True, save_path="./imputegap/assets")




.. _parameterization:

Parameterization
----------------

ImputeGAP provides optimization techniques that automatically identify the optimal hyperparameters for a specific algorithm in relation to a given dataset.
The available optimizers are: Greedy Optimizer (GO), Bayesian Optimizer (BO), Particle Swarm Optimizer (PSO), and Successive Halving (SH).

**Example Auto-ML**

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("eeg-alcohol"))
    ts_1.normalize(normalizer="min_max")

    # 3. contamination of the data
    ts_mask = ts_1.Contamination.mcar(ts_1.data)

    # 4. imputation of the contaminated data
    # imputation with AutoML which will discover the optimal hyperparameters for your dataset and your algorithm
    imputer = Imputation.MatrixCompletion.CDRec(ts_mask).impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "ray_tune"})

    # 5. score the imputation with the raw_data
    imputer.score(ts_1.data, imputer.recov_data)

    # 6. display the results
    ts_1.print_results(imputer.metrics)
    ts_1.plot(input_data=ts_1.data, incomp_data=ts_mask, recov_data=imputer.recov_data, max_series=9, subplot=True, save_path="./imputegap/assets", display=True)

    # 7. save hyperparameters
    utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg", optimizer="ray_tune")




.. _explainer:

Explainer
---------


ImputeGAP allows users to explore the features in the data that impact the imputation results
through Shapely Additive exPlanations ([**SHAP**](https://shap.readthedocs.io/en/latest/)). To attribute a meaningful interpretation of the SHAP results, ImputeGAP groups the extracted features into four categories:
geometry, transformation, correlation, and trend.


**Example Explainer**

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.recovery.explainer import Explainer
    from imputegap.tools import utils

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("eeg-alcohol"))

    # 3. call the explanation of your dataset with a specific algorithm to gain insight on the Imputation results
    shap_values, shap_details = Explainer.shap_explainer(input_data=ts_1.data, extractor="pycatch", pattern="mcar",
                                                         missing_rate=0.25, limit_ratio=1, split_ratio=0.7,
                                                         file_name="eeg-alcohol", algorithm="cdrec")

    # [OPTIONAL] print the results with the impact of each feature.
    Explainer.print(shap_values, shap_details)




.. _downstream:

Downstream
----------


ImputeGAP is a versatile library designed to help users evaluate both the upstream aspects (e.g., errors, entropy, correlation) and the downstream impacts of data imputation. By leveraging a built-in Forecaster, users can assess how the imputation process influences the performance of specific tasks.

**Example Downstream**

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("chlorine"))
    ts_1.normalize(normalizer="min_max")

    # 3. contamination of the data
    ts_mask = ts_1.Contamination.missing_percentage(ts_1.data, rate_series=0.8)
    ts_2 = TimeSeries().import_matrix(ts_mask)

    # 4. imputation of the contaminated data
    imputer = Imputation.MatrixCompletion.CDRec(ts_2.data)
    imputer.impute()

    # [OPTIONAL] save your results in a new Time Series object
    ts_3 = TimeSeries().import_matrix(imputer.recov_data)

    # 5. score the imputation with the raw_data
    downstream_options = {"evaluator": "forecaster", "model": "prophet"}
    imputer.score(ts_1.data, ts_3.data)  # upstream standard analysis
    imputer.score(ts_1.data, ts_3.data, downstream=downstream_options)  # downstream advanced analysis

    # 6. display the results
    ts_3.print_results(imputer.metrics, algorithm=imputer.algorithm)
    ts_3.print_results(imputer.downstream_metrics, algorithm=imputer.algorithm)




.. _benchmark:

Benchmark
---------


ImputeGAP enables users to comprehensively evaluate the efficiency of algorithms across various datasets.


**Example Benchmark**

.. code-block:: python

    from imputegap.recovery.benchmark import Benchmark

    # VARIABLES
    save_dir = "./analysis"
    nbr_run = 2

    # SELECT YOUR DATASET(S) :
    datasets_demo = ["eeg-alcohol", "eeg-reading"]

    # SELECT YOUR OPTIMIZER :
    optimiser_bayesian = {"optimizer": "bayesian", "options": {"n_calls": 15, "n_random_starts": 50, "acq_func": "gp_hedge", "metrics": "RMSE"}}
    optimizers_demo = [optimiser_bayesian]

    # SELECT YOUR ALGORITHM(S) :
    algorithms_demo = ["mean", "cdrec", "stmvl", "iim", "mrnn"]

    # SELECT YOUR CONTAMINATION PATTERN(S) :
    patterns_demo = ["mcar"]

    # SELECT YOUR MISSING RATE(S) :
    x_axis = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    # START THE ANALYSIS
    list_results, sum_scores = Benchmark().eval(algorithms=algorithms_demo, datasets=datasets_demo, patterns=patterns_demo, x_axis=x_axis, optimizers=optimizers_demo, save_dir=save_dir, runs=nbr_run)





