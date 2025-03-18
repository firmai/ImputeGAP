=========
Tutorials
=========

.. _loading-preprocessing:

Loading and Preprocessing
-------------------------

ImputeGAP comes with several time series datasets. You can find them inside the submodule ``ts.datasets``.

As an example, we start by using eeg-alcohol, a standard dataset composed of individuals with a genetic predisposition to
alcoholism. The dataset contains measurements from 64 electrodes placed on subject’s scalps, sampled at 256 Hz (3.9-ms epoch) for 1 second. The dimensions of the dataset are 64 series, each containing 256 values.


.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the TimeSeries()
    ts = TimeSeries()
    print(f"ImputeGAP datasets : {ts.datasets}")

    # load the timeseries from file or from the code
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # plot a subset of time series
    ts.plot(input_data=ts.data, nbr_series=9, nbr_val=100, save_path="./imputegap_assets")

    # print a subset of time series
    ts.print(nbr_series=6, nbr_val=20)


.. _contamination:

Contamination
-------------
We now describe how to simulate missing values in the loaded dataset. ImputeGAP implements eight different missingness patterns. You can find them inside the module ``ts.patterns``.

As example, we show how to contaminate the eeg-alcohol dataset with the MCAR pattern:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the TimeSeries() object
    ts = TimeSeries()
    print(f"Missingness patterns : {ts.patterns}")

    # load and normalize the timeseries
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # contaminate the time series with MCAR pattern
    ts_m = ts.Contamination.missing_completely_at_random(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

    # plot the contaminated time series
    ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets")



If you need to remove data following a specific distribution, please refer to this `tutorial <tutorials_distribution.html>`_.



.. _imputation:

Imputation
----------

In this section, we will illustrate how to impute the contaminated time series. Our library implements five families of imputation algorithms. Statistical, Machine Learning, Matrix Completion, Deep Learning, and Pattern Search Methods.
You can find the list of algorithms inside the module ``ts.algorithms``.

Imputation can be performed using either default values or user-defined values. To specify the parameters, please use a dictionary in the following format:

.. code-block:: python
    params = {"param_1": 42.1, "param_2": "some_string", "params_3": True}



Let's illustrate the imputation using the CDRec Algorithm from the Matrix Completion family.

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the TimeSeries() object
    ts = TimeSeries()
    print(f"Imputation algorithms : {ts.algorithms}")

    # load and normalize the timeseries
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # contaminate the time series
    ts_m = ts.Contamination.missing_completely_at_random(ts.data)

    # impute the contaminated series
    imputer = Imputation.MatrixCompletion.CDRec(ts_m)
    imputer.impute()

    # compute and print the imputation metrics
    imputer.score(ts.data, imputer.recov_data)
    ts.print_results(imputer.metrics)

    # plot the recovered time series
    ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, save_path="./imputegap_assets")




.. _parameterization:

Parameterization
----------------

The Optimizer component manages algorithm configuration and hyperparameter tuning. To invoke the tuning process, users need to specify the optimization option during the Impute call by selecting the appropriate input for the algorithm. The parameters are defined by providing a dictionary containing the ground truth, the chosen optimizer, and the optimizer's options. Several search algorithms are available, including those provided by (`Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_).

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the TimeSeries() object
    ts = TimeSeries()
    print(f"AutoML Optimizers : {ts.optimizers}")

    # load and normalize the timeseries
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # contaminate and impute the time series
    ts_m = ts.Contamination.missing_completely_at_random(ts.data)
    imputer = Imputation.MatrixCompletion.CDRec(ts_m)

    # use Ray Tune to fine tune the imputation algorithm
    imputer.impute(user_def=False, params={"input_data": ts.data, "optimizer": "ray_tune"})

    # compute and print the imputation metrics
    imputer.score(ts.data, imputer.recov_data)
    ts.print_results(imputer.metrics)

    # plot the recovered time series
    ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, save_path="./imputegap_assets", display=True)

    # save hyperparameters
    utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg-alcohol", optimizer="ray_tune")




