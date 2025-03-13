
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

from scipy.stats import norm
import numpy as np

for series in ts.data:
    N = len(series)
    P = int(N*0.1)
    R = np.arange(P, N)

    mean = np.mean(ts.data)
    probabilities = norm.pdf(R, loc=P + mean * (N - P), scale=0.1 * (N - P))
    probabilities /= probabilities.sum()
ts_m2 = ts.Contamination.distribution(ts.data, rate_dataset=0.2, rate_series=0.4, probabilities=probabilities, seed=True)

# plot the contaminated time series
ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap/assets")
ts.plot(ts.data, ts_m2, nbr_series=9, subplot=True, save_path="./imputegap/assets")