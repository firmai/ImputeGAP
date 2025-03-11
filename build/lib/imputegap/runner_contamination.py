from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("eeg-alcohol"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data with MCAR pattern
incomp_data = ts_1.Contamination.mcar(ts_1.data, rate_dataset=0.2, rate_series=0.2, seed=True)

# [OPTIONAL] you can plot your raw data / print the contamination
ts_1.print(nbr_val=12, nbr_series=7)
ts_1.plot(ts_1.data, incomp_data, nbr_series=9, subplot=True, save_path="./imputegap/assets")
