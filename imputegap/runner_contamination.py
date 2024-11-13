from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("eeg-alcohol"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data with MCAR scenario
infected_data = ts_1.Contaminate.mcar(ts_1.data, series_impacted=0.4, missing_rate=0.2, use_seed=True)

# [OPTIONAL] you can plot your raw data / print the contamination
ts_1.print(limit=10)
ts_1.plot(ts_1.data, infected_data, title="contamination", max_series=1, save_path="./assets")