from imputegap.recovery.contamination import Contamination
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.optimization import Optimization
from imputegap.tools.utils import display_title

import os


def check_block_size(filename):
    if "test" in filename:
        return (2, 2)
    else:
        return (10, 10)


if __name__ == '__main__':

    display_title()
    datasets = ["bafu", "chlorine", "climate", "drift", "egg", "meteo", "test-large"]

    for filename in datasets :
        file_path = os.path.join("./dataset/", filename + ".txt")
        gap = TimeSeries(data=file_path)

        block_size, plot_limit = check_block_size(filename)

        gap.print(limitation=5)
        gap.plot(title="test", save_path="assets", limit=6, display=False)

        gap.ts_contaminate = Contamination.mcar(ts=gap.data, series_impacted=0.4, missing_rate=0.4, block_size=block_size, protection=0.1, use_seed=True, seed=42)
        gap.print(limitation=5)
        gap.plot(ts_type="contamination", title="test", save_path="assets", limit=3, display=False)

        for algo in ["cdrec", "stmvl", "iim", "mrnn"]:
            print("RUN OPTIMIZATION FOR : ", algo, "... with ", filename, "...")
            optimal_params, yi = Optimization.Bayesian.bayesian_optimization(ground_truth=gap.data, contamination=gap.ts_contaminate, algorithm=algo, n_calls=100)
            print("\nOptical Params : ", optimal_params)
            print("\nyi : ", yi, "\n")
            Optimization.save_optimization(optimal_params=optimal_params, algorithm=algo, dataset=filename, optimizer="b")
            print("\n", "_"*95, "end")
        print("\n", "_" * 95, "end")
    print("\n", "_" * 95, "end")