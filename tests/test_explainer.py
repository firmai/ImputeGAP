import unittest
import numpy as np
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestExplainer(unittest.TestCase):

    def test_explainer_shap(self):
        """
        Verify if the SHAP explainer is working
        """
        filename = "chlorine"

        RMSE = [0.2780398282009316, 0.09024192407753003, 0.06264295609967269, 0.06008285096152065, 0.04622875362843607,
         0.04100194460489083, 0.03182402833276032, 0.04031085927584528, 0.08353853381025556, 0.08183653114000404,
         0.0712546131146801, 0.07127388277211984, 0.07853099688546698, 0.06457276731357126, 0.056051361732355906]

        expected_categories, expected_features, _ = Explainer.load_configuration()

        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path(filename))

        shap_values, shap_details = Explainer.shap_explainer(input_data=ts_1.data, file_name=filename, rate_dataset=0.3,
                                                             seed=True, verbose=True)

        self.assertTrue(shap_values is not None)
        self.assertTrue(shap_details is not None)

        for i, (_, output) in enumerate(shap_details):
            assert np.isclose(RMSE[i], output, atol=0.01)

        for i, (x, algo, rate, description, feature, category, mean_features) in enumerate(shap_values):
            assert rate >= 0, f"Rate must be >= 0, but got {rate}"

            self.assertTrue(x is not None and not (isinstance(x, (int, float)) and np.isnan(x)))
            self.assertTrue(algo is not None)
            self.assertTrue(rate is not None and not (isinstance(rate, (int, float)) and np.isnan(rate)))
            self.assertTrue(description is not None)
            self.assertTrue(feature is not None)
            self.assertTrue(category is not None)
            self.assertTrue(mean_features is not None and not (isinstance(mean_features, (int, float)) and np.isnan(mean_features)))

            # Check relation feature/category
            feature_found_in_category = False
            for exp_category, exp_features in expected_categories.items():
                if feature in exp_features:
                    assert category == exp_category, f"Feature '{feature}' must in '{exp_category}', but is in '{category}'"
                    feature_found_in_category = True
                    break
            assert feature_found_in_category, f"Feature '{feature}' not found in any category"

            # Check relation description/feature
            if feature in expected_features:
                expected_description = expected_features[feature]
                assert description == expected_description, f"Feature '{feature}' has wrong description. Expected '{expected_description}', got '{description}' "
            else:
                assert False, f"Feature '{feature}'not found in the FEATURES dictionary"
