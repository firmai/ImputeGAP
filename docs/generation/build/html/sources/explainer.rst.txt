=========
Explainer
=========

ImputeGAP provides insights into the algorithm's behavior by identifying the features that impact the most the imputation results. It trains a regression model to predict imputation results across various methods and uses SHapley Additive exPlanations (`SHAP <https://shap.readthedocs.io/en/latest/>`_) to reveal how different time series features influence the model’s predictions.

Let's illustrate the explainer using the CDRec Algorithm and MCAR missingness pattern:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.recovery.explainer import Explainer
    from imputegap.tools import utils

    # initialize the TimeSeries() object
    ts = TimeSeries()

    # load and normalize the timeseries
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # configure the explanation
    shap_values, shap_details = Explainer.shap_explainer(input_data=ts.data, extractor="pycatch", pattern="missing_completely_at_random", file_name=ts.name, algorithm="CDRec")

    # print the impact of each feature
    Explainer.print(shap_values, shap_details)




