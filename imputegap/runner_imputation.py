from imputegap.recovery.contamination import Contamination
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.optimization import Optimization
from imputegap.tools.utils import display_title
from imputegap.tools import utils
import os


def check_block_size(filename):
    if "test" in filename:
        return (2, 2)
    else:
        return (10, 1)


if __name__ == '__main__':

    display_title()

    filename = "eeg"
    load = False
    algo = "cdrec"

    file_path = os.path.join("./dataset/", filename + ".txt")
    gap = TimeSeries(data=file_path, normalization="z_score")

    block_size, plot_limit = check_block_size(filename)

    gap.print(limitation=5)
    gap.plot(ts_type="gt", title="test", save_path="assets", limitation=0, display=False)

    gap.ts_contaminate = Contamination.scenario_mcar(ts=gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=block_size, protection=0.1, use_seed=True, seed=42)
    gap.print(limitation=10)
    gap.plot(ts_type="contamination", title="test", save_path="assets", limitation=plot_limit, display=False)

    if load:
        gap.optimal_params = utils.load_parameters(query="optimal", algorithm=algo)
    else:
        optimal_params, yi = Optimization.Bayesian.bayesian_optimization(ground_truth=gap.ts,
                                                                         contamination=gap.ts_contaminate,
                                                                         algorithm=algo, n_calls=3)
        gap.optimal_params = tuple(optimal_params.values())
        print("\nOptical Params : ", gap.optimal_params)

    gap.ts_imputation, gap.metrics = Imputation.MR.cdrec(ground_truth=gap.ts, contamination=gap.ts_contaminate, params=gap.optimal_params)
    gap.print(limitation=10)
    gap.print_results()

    gap.plot(ts_type="imputation", title="test", save_path="assets", limitation=plot_limit, display=False)

    print("\n", "_"*95, "end")