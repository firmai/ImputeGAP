from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("eeg-alcohol"))

# 3. call the explanation of your dataset with a specific algorithm to gain insight on the Imputation results
shap_values, shap_details = Explainer.shap_explainer(input_data=ts_1.data, algorithm="cdrec", extractor="pycatch22", pattern="mcar", missing_rate=0.25, limit_ratio=1, split_ratio=0.7, file_name="eeg-alcohol")

# [OPTIONAL] print the results with the impact of each feature.
Explainer.print(shap_values, shap_details)