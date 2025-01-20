from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("eeg-alcohol"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, dataset_rate=0.4, series_rate=0.4, block_size=10, offset=0.1, seed=True)

# [OPTIONAL] save your results in a new Time Series object
ts_2 = TimeSeries().import_matrix(incomp_data)

# 4. imputation of the contaminated data
# choice of the algorithm, and their parameters (default, automl, or defined by the user)
test = Imputation.DeepLearning.PRISTI(ts_2.data)

# imputation with default values
#test.impute()

test.impute(params={"target_strategy": "hybrid", "unconditional": True, "seed": 42, "device": "cpu"})
# OR imputation with user defined values
# >>> cdrec.impute(params={"rank": 5, "epsilon": 0.01, "iterations": 100})

# [OPTIONAL] save your results in a new Time Series object
ts_3 = TimeSeries().import_matrix(test.recov_data)

ts_3.print()

# 5. score the imputation with the raw_data
test.score(ts_1.data, ts_3.data)

# 6. display the results
ts_3.print_results(test.metrics, algorithm="xxx")

ts_3.plot(input_data=ts_1.data, incomp_data=ts_2.data, recov_data=ts_3.data, max_series=9, subplot=True, save_path="./assets")
