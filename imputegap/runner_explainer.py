from imputegap.recovery.manager import TimeSeries
from imputegap.explainer.explainer import Explainer
from imputegap.tools.utils import display_title

import os


def check_block_size(filename):
    if "test" in filename:
        return (2, 2)
    else:
        return (10, 10)


if __name__ == '__main__':

    display_title()

    filename = "chlorine"
    file_path = os.path.join("./dataset/", filename + ".txt")
    gap = TimeSeries(data=file_path)

    shap_values, shap_details = Explainer.shap_explainer(ground_truth=gap.ts, file_name=filename)

    Explainer.print(shap_values, shap_details)


    print("\n", "_"*95, "end")