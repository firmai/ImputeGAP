import numpy as np

from imputegap.recovery.manager import TimeSeries

import os

from imputegap.tools import utils



if __name__ == '__main__':

    dataset = "eeg"

    utils.display_title()

    ts_1 = TimeSeries()
    ts_1.load_timeseries(data=utils.search_path(dataset))
    ts_1.print(view_by_series=True)
    ts_1.plot(ts_1.data, display=False)
    infected_matrix = ts_1.Contaminate.mcar(ts=ts_1.data, use_seed=True, seed=42)

    ts_2 = TimeSeries()
    ts_2.import_matrix(infected_matrix)

    ts_2.print(view_by_series=True)
    ts_2.plot(ts_1.data, ts_2.data, max_series=1, save_path="assets")

    print("\n", "_"*95, "end")