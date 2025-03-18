from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()
print(f"ImputeGAP downstream models for forcasting : {ts.downstream_models}")

# load and normalize the timeseries
ts.load_series(utils.search_path("forecast-economy"))
ts.normalize(normalizer="min_max")

# contaminate the time series
ts_m = ts.Contamination.missing_percentage(ts.data, rate_series=0.8)

# define and impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# compute print the downstream results
downstream_config = {"task": "forecast", "model": "hw-add"}
imputer.score(ts.data, imputer.recov_data, downstream=downstream_config)
ts.print_results(imputer.downstream_metrics, algorithm=imputer.algorithm)