from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("chlorine"))

# 3. call the explanation of your dataset with a specific algorithm to gain insight on the Imputation results
shap_values, shap_details = Explainer.shap_explainer(raw_data=ts_1.data, missing_rate=0.25, limitation=50, splitter=35, file_name="chlorine", algorithm="cdrec")

# [OPTIONAL] print the results with the impact of each feature.
Explainer.print(shap_values, shap_details)