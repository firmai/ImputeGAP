from imputegap.contamination.contamination import Contamination
from imputegap.imputation.imputation import Imputation
from imputegap.manager.manager import TimeSeries

def display_title(title="Master Thesis", aut="Quentin Nater", lib="ImputeGAP",
                  university="University Fribourg - exascale infolab", file="runner"):
    print("=" * 100)
    print(f"{title} : {aut}")
    print("=" * 100)
    print(f"    {lib} - {university}")
    print("=" * 100)
    print(f"    {file}.py")
    print("=" * 100)


if __name__ == '__main__':

    display_title()

    gap = TimeSeries(data="./dataset/test.txt")

    """
    X = [[1.5, 0.5, 0.1, 0.1, 0.1, 0.1, 1.4, 1.0, 1.0, 0.0],
        [2.5, 0.2, 2.9, 2.0, 1.8, 1.8, 2.4, 1.0, 1.0, 0.0],
        [1.5, 0.3, 2.8, 1.9, 1.7, 1.9, 1.4, 1.0, 1.0, 1.0],
        [2.5, 0.4, 2.7, 1.8, 1.6, 1.6, 2.5, 1.0, 1.0, 1.0],
        [1.5, 0.9, 2.6, 1.7, 1.5, 0.8, 1.0, 1.0, 1.0, 0.5],
        [2.0, 1.0, 2.5, 1.6, 1.4, 1.8, 1.0, 1.0, 1.8, 1.5],
        [2.0, 0.0, 2.4, 1.5, 1.3, 1.9, 2.0, 1.0, 1.9, 0.6],
        [2.0, 0.0, 2.3, 1.4, 1.2, 1.1, 2.0, 1.0, 2.0, 1.4],
        [2.0, 0.0, 2.1, 1.3, 1.1, 1.0, 1.0, 2.0, 2.1, 0.5],
        [2.0, 0.0, 2.0, 1.2, 1.0, 1.9, 1.1, 0.0, 2.0, 1.5],
        [2.0, 0.0, 1.9, 1.1, 0.9, 0.1, 1.5, 1.0, 1.9, 0.7],
        [1.0, 0.0, 1.8, 1.0, 0.8, 0.2, 1.4, 1.0, 1.8, 0.8],
        [1.0, 0.0, 1.7, 0.9, 0.7, 0.6, 1.7, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.6, 0.8, 0.6, 0.7, 1.8, 1.0, 1.0, 1.5],
        [1.0, 1.0, 1.5, 0.7, 0.5, 0.8, 1.9, 1.0, 1.0, 1.9],
        [2.0, 1.0, 1.3, 0.6, 0.4, 0.9, 2.0, 1.0, 1.0, 1.1],
        [1.0, 1.0, 1.2, 0.5, 0.3, 0.4, 0.2, 1.0, 1.0, 1.5],
        [2.0, 1.0, 1.1, 0.4, 0.2, 0.1, 0.1, 1.0, 1.0, 1.4],
        [1.0, 1.0, 1.0, 0.3, 0.1, 0.2, 0.1, 1.0, 1.0, 1.6],
        [2.0, 1.0, 0.9, 0.2, 0.0, 0.3, -0.1, 1.0, 1.0, 1.5],
        [2.0, 1.0, 0.9, 0.2, 0.0, 0.3, -0.1, 1.0, 1.0, 1.5],
        [1.8, 0.8, 0.7, 0.0, -0.2, 0.1, -0.3, 0.8, 0.8, 1.4],
        [2.2, 1.2, 1.1, 0.4, 0.2, 0.5, 0.1, 1.2, 1.2, 1.5],
        [2.0, 1.0, 1.0, 0.2, 0.0, 0.3, -0.1, 1.0, 1.0, 1.3],
        [2.1, 1.1, 1.1, 0.1, 0.1, 0.2, 0.0, 1.1, 1.1, 1.2]]
    gap = TimeSeries(data=X)
    """

    gap.print(limitation=5)
    gap.plot(title="test", save_path="assets/", limitation=6, display=False)

    gap.ts_contaminate = Contamination.scenario_mcar(ts=gap.ts, series_impacted=0.4, missing_rate=0.4, block_size=2, protection=0.1, use_seed=True, seed=42)
    gap.print()
    gap.plot(ts_type="contamination", title="test", save_path="assets/", limitation=3, display=False)

    #gap.ts_imputation, gap.metrics = Imputation.Stats.zero_impute(ground_truth=gap.ts, contamination=gap.ts_contaminate)
    #gap.print()
    #gap.print_results()
    #gap.plot(ts_type="imputation", title="test", save_path="assets/", limitation=2, display=True)

    print("\n", "_"*95, "end")