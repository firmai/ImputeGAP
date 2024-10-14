from imputegap.recovery.manager import TimeSeries
from imputegap.explainer.explainer import Explainer
from imputegap.tools import utils
from imputegap.tools.utils import display_title


if __name__ == '__main__':

    dataset = "eeg"
    display_title()

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_timeseries(utils.search_path(dataset))

    # call the explanation of your dataset with a specific algorithm to gain insight on the Imputation results
    shap_values, shap_details = Explainer.shap_explainer(raw_data=ts_1.data, file_name=dataset, algorithm="iim")

    # [OPTIONAL] print the results with the impact of each feature.
    Explainer.print(shap_values, shap_details)