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
        return (10, 10)


if __name__ == '__main__':

    display_title()

    filename = "eeg"

    file_path = os.path.join("./dataset/", filename + ".txt")
    gap = TimeSeries(data=file_path, normalization="z_score", limitation_values=100)

    block_size, plot_limit = check_block_size(filename)

    gap.ts_contaminate = Contamination.scenario_missing_percentage(ts=gap.ts, series_impacted=0.1, missing_rate=0.4, protection=0.1, use_seed=True, seed=42)
    gap.print(limitation=10)
    gap.plot(ts_type="contamination", title="test", save_path="assets", limitation=plot_limit, display=True)

    print("\n", "_"*95, "end")