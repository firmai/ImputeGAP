from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()
print(f"Missingness patterns : {ts.patterns}")

# load and normalize the timeseries
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# contaminate the time series with MCAR pattern
ts_m = ts.Contamination.missing_completely_at_random(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

# plot the contaminated time series
ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets")