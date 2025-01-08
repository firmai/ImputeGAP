from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("eeg-alcohol"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
incomp_data = ts_1.Contamination.mcar(ts_1.data)

# [OPTIONAL] save your results in a new Time Series object
ts_2 = TimeSeries().import_matrix(incomp_data)

# 4. imputation of the contaminated data
# choice of the algorithm, and their parameters (default, automl, or defined by the user)
cdrec = Imputation.MatrixCompletion.CDRec(ts_2.data)

# imputation with default values
cdrec.impute()
# OR imputation with user defined values
# >>> cdrec.impute(params={"rank": 5, "epsilon": 0.01, "iterations": 100})

# [OPTIONAL] save your results in a new Time Series object
ts_3 = TimeSeries().import_matrix(cdrec.recov_data)

# 5. score the imputation with the raw_data
cdrec.score(ts_1.data, ts_3.data)

# 6. display the results
ts_3.print_results(cdrec.metrics, algorithm="cdrec")
ts_3.plot(input_data=ts_1.data, incomp_data=ts_2.data, recov_data=ts_3.data, subplot=True, save_path="./assets")
