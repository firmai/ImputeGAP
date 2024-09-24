from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation
from imputegap.manager.manager import TimeSeries

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

    filename = "./dataset/test.txt"
    gap = TimeSeries(data=filename)

    block_size, plot_limit = check_block_size(filename)

    gap.print(limitation=5)
    gap.plot(title="test", save_path="assets/", limitation=6, display=False)

    gap.ts_contaminate = Contamination.scenario_mcar(ts=gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=block_size, protection=0.1, use_seed=True, seed=42)
    gap.print()
    gap.plot(ts_type="contamination", title="test", save_path="assets/", limitation=3, display=False)

    gap.ts_imputation, gap.metrics = Imputation.Pattern.stmvl_imputation(ground_truth=gap.ts, contamination=gap.ts_contaminate)
    gap.print()
    gap.print_results()
    gap.plot(ts_type="imputation", title="test", save_path="assets/", limitation=plot_limit, display=True)

    print("\n", "_"*95, "end")