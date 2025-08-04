import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils


class TestDownstream(unittest.TestCase):


    def test_downstream(self):
        """
        Verify if the downstream process is working
        """
        # Load and normalize the series
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("forecast-economy"))
        ts_1.normalize(normalizer="min_max")

        # Create a mask for contamination
        ts_mask = ts_1.Contamination.mcar(ts_1.data, rate_dataset=0.2, rate_series=0.8)

        # Perform imputation
        imputer = Imputation.MatrixCompletion.CDRec(ts_mask)
        imputer.impute()

        # Configure downstream options
        downstream_options = [{"task": "forecast", "model": "prophet", "params": None, "plots": False},
                              {"task": "forecast", "model": "naive", "params": None, "plots": False},
                              {"task": "forecast", "model": "exp-smoothing", "params": None, "plots": False},
                              {"task": "forecast", "model": "nbeats", "params": None, "plots": False}]

        for options in downstream_options:
            model = options.get("model")

            # Score and evaluate
            imputer.score(ts_1.data, imputer.recov_data)
            imputer.score(ts_1.data, imputer.recov_data, downstream=options)

            # Assert metrics are dictionaries with values
            self.assertIsInstance(imputer.metrics, dict, "imputer.metrics should be a dictionary, for " + model)
            self.assertTrue(imputer.metrics, "imputer.metrics should not be empty, for " + model)

            self.assertIsInstance(imputer.downstream_metrics, dict, "imputer.downstream_metrics should be a dictionary, for " + model)
            self.assertTrue(imputer.downstream_metrics, "imputer.downstream_metrics should not be empty, for " + model)

            # Display the results
            ts_1.print_results(imputer.metrics, algorithm=model)
            ts_1.print_results(imputer.downstream_metrics, algorithm=model)