from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initiate the TimeSeries() object that will stay with you throughout the analysis
ts = TimeSeries()
print(f"ImputeGAP datasets : {ts.datasets}")

# load the timeseries from file or from the code
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# [OPTIONAL] plot a subset of time series
ts.plot(input_data=ts.data, nbr_series=6, nbr_val=100, save_path="./imputegap/assets")

# [OPTIONAL] print a subset of time series
ts.print(nbr_series=3,  nbr_val=20)
