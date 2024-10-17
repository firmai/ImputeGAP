from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("eeg-test"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
infected_data = ts_1.Contaminate.missing_percentage(ts_1.data)

# [OPTIONAL] save your results in a new Time Series object
ts_2 = TimeSeries().import_matrix(infected_data)

# 4. imputation of the contaminated data
# choice of the algorithm, and their parameters (default, automl, or defined by the user)
cdrec = Imputation.MD.CDRec(ts_2.data)

# imputation with default values
cdrec.impute()
# OR imputation with user defined values
# >>> cdrec.impute(params={"rank": 5, "epsilon": 0.01, "iterations": 100})

# [OPTIONAL] save your results in a new Time Series object
ts_3 = TimeSeries().import_matrix(cdrec.imputed_matrix)

# 5. score the imputation with the raw_data
cdrec.score(ts_1.data, ts_3.data)

# 6. display the results
ts_3.print_results(cdrec.metrics, algorithm="cdrec")
ts_3.plot(raw_data=ts_1.data, infected_data=ts_2.data, imputed_data=ts_3.data, title="imputation", max_series=1, save_path="./assets", display=True)
