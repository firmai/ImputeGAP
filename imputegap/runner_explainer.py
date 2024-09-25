from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation
from imputegap.manager.manager import TimeSeries
from imputegap.explainer.explainer import Explainer
import os


def display_title(title="Master Thesis", aut="Quentin Nater", lib="ImputeGAP", university="University Fribourg - exascale infolab"):
    print("=" * 100)
    print(f"{title} : {aut}")
    print("=" * 100)
    print(f"    {lib} - {university}")
    print("=" * 100)


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