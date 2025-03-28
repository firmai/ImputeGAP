=====================
Downstream Evaluation
=====================

ImputeGAP includes a dedicated module for systematically evaluating the impact of data imputation on downstream tasks. Currently, forecasting is the primary supported task, with plans to expand to additional tasks in the future.





Below is an example of how to call the downstream process for the model by defining a dictionary with the task and the name the model:

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the timeseries
    ts.load_series(utils.search_path("forecast-economy"))
    ts.normalize(normalizer="min_max")

    # contaminate the time series
    ts_m = ts.Contamination.aligned(ts.data, rate_series=0.8)

    # define and impute the contaminated series
    imputer = Imputation.MatrixCompletion.CDRec(ts_m)
    imputer.impute()

    # compute and print the downstream results
    downstream_config = {"task": "forecast", "model": "hw-add", "comparator": "ZeroImpute"}
    imputer.score(ts.data, imputer.recov_data, downstream=downstream_config)
    ts.print_results(imputer.downstream_metrics, algorithm=imputer.algorithm)




To list all the available downstream models, you can use this command:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP downstream models for forcasting : {ts.forecasting_models}")

