from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts_origin = TimeSeries()
ts_origin.load_series(utils.search_path("airq"))

ts_cont = TimeSeries()
ts_cont.load_series(utils.search_path("airq_m200.txt"))
ts_m = ts_cont.data

# impute the contaminated series
imputer = Imputation.DeepLearning.BayOTIDE(incomp_data=ts_m)
imputer.impute()

# compute and print the imputation metrics
imputer.score(ts_origin.data, imputer.recov_data)
ts_origin.print_results(imputer.metrics)

# plot the recovered time series
ts_origin.plot(input_data=ts_origin.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")