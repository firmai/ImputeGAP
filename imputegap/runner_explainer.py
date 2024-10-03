from imputegap.recovery.manager import TimeSeries
from imputegap.explainer.explainer import Explainer
from imputegap.tools import utils
from imputegap.tools.utils import display_title

import os


def check_block_size(filename):
    if "test" in filename:
        return (2, 2)
    else:
        return (10, 10)


if __name__ == '__main__':

    dataset = "eeg"
    display_title()

    ts_1 = TimeSeries()
    ts_1.load_timeseries(utils.search_path(dataset))
    shap_values, shap_details = Explainer.shap_explainer(raw_data=ts_1.data, file_name=dataset)

    Explainer.print(shap_values, shap_details)

    print("\n", "_"*95, "end")