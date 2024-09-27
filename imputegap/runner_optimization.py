from imputegap.contamination.contamination import Contamination
from imputegap.manager.manager import TimeSeries
import os

from imputegap.optimization.bayesian_optimization import Optimization


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

    filename = "eeg"
    file_path = os.path.join("./dataset/", filename + ".txt")
    gap = TimeSeries(data=file_path)

    block_size, plot_limit = check_block_size(filename)

    gap.print(limitation=5)
    gap.plot(title="test", save_path="assets", limitation=6, display=False)

    gap.ts_contaminate = Contamination.scenario_mcar(ts=gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=block_size, protection=0.1, use_seed=True, seed=42)
    gap.print()
    gap.plot(ts_type="contamination", title="test", save_path="assets", limitation=3, display=False)

    optimal_params, yi = Optimization.Bayesian.bayesian_optimization(ground_truth=gap.ts, contamination=gap.ts_contaminate, algorithm="iim")

    print("\nOptical Params : ", optimal_params)
    print("\nyi : ", yi, "\n")

    Optimization.save_optimization(optimal_params=optimal_params, algorithm="cdrec")

    print("\n", "_"*95, "end")