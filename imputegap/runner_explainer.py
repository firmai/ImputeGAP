from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()

# load and normalize the timeseries
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# explanation of the imputation with a specific algorithm, pattern of contamination and dataset
shap_values, shap_details = Explainer.shap_explainer(input_data=ts.data, extractor="pycatch", pattern="mcar",
                                                     missing_rate=0.25, limit_ratio=1, split_ratio=0.7,
                                                     file_name="eeg-alcohol", algorithm="cdrec")

# print the impact of each feature
Explainer.print(shap_values, shap_details)