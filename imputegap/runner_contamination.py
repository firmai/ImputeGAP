from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"Missingness patterns : {ts.patterns}")

# load and normalize the dataset
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# contaminate the time series with MCAR pattern
ts_m = ts.Contamination.overlap(ts.data, rate_series=0.1, limit=1, shift=0.05, offset=0.1)

# plot the contaminated time series
ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets")