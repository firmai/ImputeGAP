import unittest
import numpy as np
import math
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestContaminationMP(unittest.TestCase):

    def test_mp_selection(self):
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
                    incomp_data = ts.Contamination.missing_percentage(input_data=ts.data, dataset_rate=S, series_rate=R, offset=P)

                    n_nan = np.isnan(incomp_data).sum()
                    expected_nan_series = math.ceil(S * M)
                    expected_nan_values = int((N - int(N * P)) * R)
                    expected_nan = expected_nan_series * expected_nan_values

                    self.assertEqual(n_nan, expected_nan, f"Expected {expected_nan} contaminated series but found {n_nan}")

    def test_mp_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("test"))

        series_impacted = [0.4, 0.8]
        missing_rates = [0.1, 0.4, 0.6]
        ten_percent_index = int(ts_1.data.shape[1] * 0.1)

        for series_sel in series_impacted:
            for missing_rate in missing_rates:

                ts_contaminate = ts_1.Contamination.missing_percentage(input_data=ts_1.data,
                                                                       dataset_rate=series_sel,
                                                                       series_rate=missing_rate, offset=0.1)

                if np.isnan(ts_contaminate[:, :ten_percent_index]).any():
                    check_position = False
                else:
                    check_position = True

                self.assertTrue(check_position, True)

    def test_mp_missing_percentage_total(self):
        """
        Test if the size of the missing percentage in a contaminated time series meets the expected number defined by the user.
        """
        datasets = ["drift", "chlorine", "eeg-alcohol", "fmri-objectviewing", "fmri-stoptask"]
        series_impacted = [0.4, 0.8]
        missing_rates = [0.2, 0.6]
        offset = 0.1

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.load_series(utils.search_path(dataset))
            M, N = ts_1.data.shape

            for series_sel in series_impacted:
                for missing_rate in missing_rates:
                    ts_contaminate = ts_1.Contamination.missing_percentage(input_data=ts_1.data,
                                                                           dataset_rate=series_sel,
                                                                           series_rate=missing_rate,
                                                                           offset=offset)

                    nbr_series_contaminated = 0
                    for current_series in ts_contaminate:

                        if np.isnan(current_series).any():
                            nbr_series_contaminated = nbr_series_contaminated+1

                            num_missing_values = np.isnan(current_series).sum()
                            expected_num_missing = int((N-int(N*offset)) * missing_rate)

                            print("\t\tNUMBR OF VALUES : ", num_missing_values)
                            print("\t\tEXPECTED VALUES : ", expected_num_missing, "\n")

                            self.assertEqual(num_missing_values, expected_num_missing,
                                msg=f"Dataset '{dataset}', Series Index {current_series}: "
                                    f"Expected {expected_num_missing} missing values, but found {num_missing_values}.")

                            percentage = ((expected_num_missing/(N-int(N*offset)))*100)
                            print("\t\tPERCENTAGE VALUES : ", percentage)
                            print("\t\tEXPECTED % VALUES : ", missing_rate*100, "\n")

                            self.assertAlmostEqual(percentage, missing_rate * 100, delta=1,
                                msg=f"Dataset '{dataset}': Expected {missing_rate * 100}%, but found {percentage}%.")

                            print("\n\n\n===============================\n\n")

                    expected_nbr_series = int(np.ceil(M*series_sel))
                    self.assertEqual(
                        nbr_series_contaminated, expected_nbr_series,
                        msg=f"Dataset '{dataset}': Expected {expected_nbr_series} contaminated series, "
                            f"but found {nbr_series_contaminated}."
                    )

                    print("NUMBR OF SERIES : ", nbr_series_contaminated)
                    print("EXPECTED SERIES : ", expected_nbr_series, "\n")