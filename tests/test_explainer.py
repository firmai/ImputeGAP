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
        RMSE = [0.2683651272658591, 0.18536679458725042, 0.1509411163650527, 0.13510891276754608, 0.10669458140782341,
                0.0945575377862439, 0.06491106119902068, 0.04802705750524594, 0.12897380114704524, 0.13074286774459615,
                0.12366536487070472, 0.12559982507214026, 0.11389311467195656, 0.08676263762857882, 0.11609908197524361]

        SHAP_VAL = [87.98, 5.06, 4.74, 1.21, 0.53, 0.28, 0.11, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0]

        expected_categories, expected_features = Explainer.load_configuration()

        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path(filename))

        shap_values, shap_details = Explainer.shap_explainer(input_data=ts_1.data, file_name=filename, limit_ratio=0.3, seed=True, verbose=True)

        self.assertTrue(shap_values is not None)
        self.assertTrue(shap_details is not None)

        for i, (_, output) in enumerate(shap_details):
            assert np.isclose(RMSE[i], output, atol=0.01)

        for i, (x, algo, rate, description, feature, category, mean_features) in enumerate(shap_values):
            assert np.isclose(SHAP_VAL[i], rate, atol=3)

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
