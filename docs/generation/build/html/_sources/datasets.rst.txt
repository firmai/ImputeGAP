========
Datasets
========

The library provides a diverse collection of datasets, each exhibiting specific patterns, trends, fields, challenges, or exceptions. These datasets can be loaded as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils
    ts_1 = TimeSeries()
    ts_1.load_series(utils.search_path("eeg-alcohol"), max_series=5, max_values=15)

You can find the complete list of available datasets at the following link: `Datasets <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/dataset>`_

Alternatively, you can import your own dataset by specifying its local path, ensuring that it meets the specific requirements of ImputeGAP.


