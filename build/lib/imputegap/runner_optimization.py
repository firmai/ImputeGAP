from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("eeg-alcohol"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
miss_matrix = ts_1.Contamination.mcar(ts_1.data)

# 4. imputation of the contaminated data
# imputation with AutoML which will discover the optimal hyperparameters for your dataset and your algorithm
cdrec = Imputation.MatrixCompletion.CDRec(miss_matrix).impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 3}})

# 5. score the imputation with the raw_data
cdrec.score(ts_1.data, cdrec.recov_data)

# 6. display the results
ts_1.print_results(cdrec.metrics)
ts_1.plot(input_data=ts_1.data, incomp_data=miss_matrix, recov_data=cdrec.recov_data, max_series=9, subplot=True, save_path="./imputegap/assets", display=True)

# 7. save hyperparameters
utils.save_optimization(optimal_params=cdrec.parameters, algorithm="cdrec", dataset="eeg", optimizer="t")