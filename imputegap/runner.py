from builtins import set

import imputegap
from imputegap.manager._manager  import TimeSeriesGAP



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

    impute_gap = TimeSeriesGAP("./dataset/chlorine.txt")
    impute_gap.print(limitation=5)
    impute_gap.plot(title="test", save_path="assets/", limitation=6, display=False)

    impute_gap.contamination_mcar(missing_rate=0.8, block_size=10, starting_position=0.1, series_selected=["0"], use_seed=True)
    impute_gap.print()
    impute_gap.plot(ts_type="contamination", title="test", save_path="assets/", limitation=2, display=True)

    print("\n", "_"*95, "end")


