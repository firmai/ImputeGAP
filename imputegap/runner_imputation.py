from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation
from imputegap.manager.manager import TimeSeries
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
        return (10, 1)


if __name__ == '__main__':

    display_title()

    filename = "eeg"
    file_path = os.path.join("./dataset/", filename + ".txt")
    gap = TimeSeries(data=file_path, normalization="z_score")

    block_size, plot_limit = check_block_size(filename)

    gap.print(limitation=5)
    gap.plot(title="test", save_path="assets", limitation=0, display=False)

    gap.ts_contaminate = Contamination.scenario_mcar(ts=gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=block_size, protection=0.1, use_seed=True, seed=42)
    gap.print()
    gap.plot(ts_type="contamination", title="test", save_path="assets", limitation=plot_limit, display=False)

    gap.optimal_params = Imputation.load_parameters(query="optimal", algorithm="cdrec")

    gap.ts_imputation, gap.metrics = Imputation.MR.cdrec(ground_truth=gap.ts, contamination=gap.ts_contaminate, params=gap.optimal_params)
    gap.print()
    gap.print_results()

    gap.plot(ts_type="imputation", title="test", save_path="assets", limitation=plot_limit, display=False)

    print("\n", "_"*95, "end")