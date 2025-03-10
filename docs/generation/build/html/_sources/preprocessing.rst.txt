==============
Pre Processing
==============

ImputeGAP includes several time series datasets, but users can also work with their own data. The library then offers various tools for cleaning and preprocessing, including normalization.

Users can first specify the number of series and values to adjust according to their needs. Then, they can choose between two normalization methods:

    - Z-score normalization: Standardizes data by subtracting the mean and dividing by the standard deviation, ensuring a mean of 0 and a standard deviation of 1.
    - Min-max normalization: Scales data to a fixed range, typically [0,1], by adjusting values based on the minimum and maximum in the dataset.

You can access the API documentation at the following link: (`normalize <imputegap.manager.html#imputegap.recovery.manager.TimeSeries.normalize>`_).

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initiate the TimeSeries() object that will stay with you throughout the analysis
    ts = TimeSeries()

    # load the timeseries from file or from the code
    ts.load_series(utils.search_path("eeg-alcohol"), nbr_series=50, nbr_val=200, )
    ts.normalize(normalizer="z_score")



