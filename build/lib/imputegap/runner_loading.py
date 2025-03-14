from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("eeg-alcohol"), nbr_series=5, nbr_val=15)
ts_1.normalize(normalizer="z_score")

# [OPTIONAL] you can plot your raw data / print the information
ts_1.plot(input_data=ts_1.data, nbr_series=10, nbr_val=100, save_path="./imputegap/assets")
ts_1.print(nbr_series=10)