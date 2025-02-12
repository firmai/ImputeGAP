from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()
ts_1.load_series(utils.search_path("airq"))
ts_1.normalize(normalizer="min_max")

incomp_data = ts_1.Contamination.missing_percentage(input_data=ts_1.data, dataset_rate=0.1)

algo = Imputation.PatternSearch.TKCM(incomp_data).impute()
algo.score(ts_1.data)
metrics = algo.metrics

ts_1.print_results(algo.metrics, algorithm=algo.algorithm)

