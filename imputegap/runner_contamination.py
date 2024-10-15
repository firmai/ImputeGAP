from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

if __name__ == '__main__':

    dataset = "eeg"
    utils.display_title()

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_timeseries(data=utils.search_path(dataset), header=True)
    ts_1.normalize(normalizer="z_score")

    # [OPTIONAL] you can plot your raw data / print the information
    ts_1.print(view_by_series=True)
    ts_1.plot(ts_1.data)


    infected_matrix = ts_1.Contaminate.mcar(ts=ts_1.data, use_seed=True, seed=2)

    # [OPTIONAL] save your results in a new Time Series object
    ts_2 = TimeSeries()
    ts_2.import_matrix(infected_matrix)

    # [OPTIONAL] you can plot your raw data / print the contamination
    ts_2.print(view_by_series=True)
    ts_2.plot(ts_1.data, ts_2.data, max_series=1, save_path="./assets")
