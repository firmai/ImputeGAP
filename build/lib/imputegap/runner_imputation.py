from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()
print(f"Imputation algorithms : {ts.algorithms}")

# load and normalize the timeseries
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# contaminate the time series
ts_m = ts.Contamination.mcar(ts.data)

# impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute() # could also use a dictionary for params: params={"rank": 5, "epsilon": 0.01, "iterations": 100}

# compute and print the imputation metrics
imputer.score(ts.data, imputer.recov_data)
ts.print_results(imputer.metrics)

# plot the recovered time series
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, save_path="./imputegap_assets")