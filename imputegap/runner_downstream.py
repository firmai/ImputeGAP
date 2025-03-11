from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()

# load and normalize the timeseries
ts.load_series(utils.search_path("chlorine"))
ts.normalize(normalizer="min_max")

# contaminate the time series
ts_m = ts.Contamination.missing_percentage(ts.data, rate_series=0.8)

# define and impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# compute and print the imputation metrics with Up and Downstream
downstream_options = {"evaluator": "forecaster", "model": "prophet"}
imputer.score(ts.data, imputer.recov_data)  # upstream standard analysis
imputer.score(ts.data, imputer.recov_data, downstream=downstream_options)  # downstream advanced analysis

# print the imputation metrics
ts.print_results(imputer.metrics, algorithm=imputer.algorithm)
ts.print_results(imputer.downstream_metrics, algorithm=imputer.algorithm)
