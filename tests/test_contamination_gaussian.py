import unittest
import numpy as np
import math
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestContamination(unittest.TestCase):

    def test_gaussian_selection(self):
        """
        the goal is to test if the number of NaN values expected are provided in the contamination output
        """

        datasets = ["drift", "chlorine", "eeg-alcohol", "fmri-objectviewing", "fmri-stoptask"]
        series_impacted = [0.1, 0.5, 1]  # percentage of series impacted
        missing_rates = [0.1, 0.5, 1]  # percentage of missing values with NaN
        P = 0.1  # offset zone

        for dataset in datasets:
            ts = TimeSeries()
            ts.load_series(utils.search_path(dataset))
            M, N = ts.data.shape  # series, values

            for S in series_impacted:
                for R in missing_rates:
                    validation = False
                    incomp_data = ts.Contamination.gaussian(input_data=ts.data, dataset_rate=S, series_rate=R, offset=P)

                    n_nan = np.isnan(incomp_data).sum()
                    expected_nan_series = math.ceil(S * M)
                    expected_nan_values = int((N - int(N * P)) * R)
                    expected_nan = expected_nan_series * expected_nan_values

                    if n_nan <= expected_nan:
                        validation = True

                    self.assertTrue(validation, f"Expected {expected_nan} contaminated series but found {n_nan}")

    def test_gaussian_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("drift"))

        series_impacted = [0.4, 0.8]
        missing_rates = [0.1, 0.4, 0.6]
        ten_percent_index = int(ts_1.data.shape[1] * 0.1)

        for series_sel in series_impacted:
            for missing_rate in missing_rates:

                ts_contaminate = ts_1.Contamination.gaussian(input_data=ts_1.data,
                                                             dataset_rate=series_sel,
                                                             series_rate=missing_rate, offset=0.1)

                if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                    check_position = False
                else:
                    check_position = True

                self.assertTrue(check_position, True)

    def test_gaussian_logic(self):
        """
        The goal is to test if the logic of the Bayesian contamination is respected.
        Specifically, contamination with a higher standard deviation should result in
        more sparsely distributed NaN values compared to a lower standard deviation.
        """

        datasets = ["chlorine"]
        nbr_series_impacted = [0.2, 0.5, 0.80]  # Percentage of series impacted
        missing_rates_per_series = [0.4, 0.6]  # Percentage of missing values with NaN
        std_devs = [0.2, 0.5]  # Standard deviations to test
        P = 0.1  # Offset zone

        for dataset in datasets:
            ts = TimeSeries()
            ts.load_series(utils.search_path(dataset))

            for S in nbr_series_impacted:
                for R in missing_rates_per_series:
                    densities = {}

                    for std_dev in std_devs:
                        # Generate contamination with the current standard deviation
                        contaminated_data = ts.Contamination.gaussian(input_data=ts.data, dataset_rate=S, series_rate=R,
                                                                      std_dev=std_dev, offset=P)

                        # Calculate positions of NaN values
                        nan_positions = np.where(np.isnan(contaminated_data))

                        # Center of the time series (considering offset zone)
                        center = int((ts.data.shape[1] + (ts.data.shape[1] * P)) // 2)

                        # Compute average distances of NaN positions from the center
                        density = np.abs(nan_positions[1] - center).mean()
                        densities[std_dev] = density

                    self.assertLess(densities[0.2], densities[0.5],
                        f"Medium deviation density {densities[0.2]} should be more tightly packed than high deviation density {densities[0.5]}, "
                        f"for dataset {dataset}, series impacted {S}, and missing rate {R}. (Center: {center})")
