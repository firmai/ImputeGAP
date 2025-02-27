from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("chlorine"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data with MCAR pattern
miss_data = ts_1.Contamination.mcar(input_data=ts_1.data, series_rate=0.2, offset=0.1)

# [OPTIONAL] you can plot your raw data / print the contamination
ts_1.print(limit_timestamps=12, limit_series=7)
ts_1.plot(ts_1.data, miss_data, max_series=-1, subplot=True, save_path="./imputegap/assets")
