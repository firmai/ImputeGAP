from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the TimeSeries()
ts = TimeSeries()
print(f"ImputeGAP datasets : {ts.datasets}")


# load the timeseries from file or from the code
ts.load_series(utils.search_path("eeg-alcohol"))

# plot a subset of time series
ts.plot(input_data=ts.data, save_path="./imputegap/assets")

# print a subset of time series
ts.print(nbr_series=6, nbr_val=20)
