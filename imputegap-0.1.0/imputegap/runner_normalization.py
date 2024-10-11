from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

if __name__ == '__main__':
    dataset, display = "chlorine", True
    utils.display_title()

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_timeseries(utils.search_path(dataset))

    # [OPTIONAL] you can plot your raw data / print the information
    ts_1.plot(raw_data=ts_1.data, title="normal", max_series=3, display=display)

    # 3. normalization of your data with z_score
    ts_1.normalize()
    ts_1.plot(raw_data=ts_1.data, title="z_score", max_series=3, display=display)

    # same process for all normalization
    ts_2 = TimeSeries()
    ts_2.load_timeseries(utils.search_path(dataset))
    ts_2.normalize(normalizer="min_max")
    ts_2.plot(raw_data=ts_2.data, title="min_max", max_series=3, display=display)


    ts_3 = TimeSeries()
    ts_3.load_timeseries(utils.search_path(dataset))
    ts_3.normalize(normalizer="m_lib")
    ts_3.plot(raw_data=ts_3.data, title="m_lib", max_series=3, display=display)


    ts_4 = TimeSeries()
    ts_4.load_timeseries(utils.search_path(dataset))
    ts_4.normalize(normalizer="z_lib")
    ts_4.plot(raw_data=ts_4.data, title="z_lib", max_series=3, display=display)


    print("\n", "_" * 95, "end")
