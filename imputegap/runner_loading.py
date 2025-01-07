from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("fmri-stoptask"), max_series=5, max_values=15)
ts_1.normalize(normalizer="z_score")

# [OPTIONAL] you can plot your raw data / print the information
ts_1.plot(input_data=ts_1.data, max_series=10, max_values=100, save_path="./imputegap/assets")
ts_1.print(limit_series=10)