from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

import numpy as np
import tsfel

ts_1 = TimeSeries()
ts_1.load_series(utils.search_path("chlorine"))

incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, dataset_rate=0.4, series_rate=0.4, block_size=10, offset=0.1, seed=True)

nan_less_matrix = incomp_data.copy()  # Copy to preserve the original matrix if needed
nan_less_matrix[np.isnan(nan_less_matrix)] = 0
M, N = nan_less_matrix.shape

cfg = tsfel.get_features_by_domain("spectral")
spectral = tsfel.time_series_features_extractor(cfg, nan_less_matrix)

cfg = tsfel.get_features_by_domain("statistical")
statistical = tsfel.time_series_features_extractor(cfg, nan_less_matrix)


cfg = tsfel.get_features_by_domain("temporal")
temporal = tsfel.time_series_features_extractor(cfg, nan_less_matrix)

cfg = tsfel.get_features_by_domain("fractal")
fractal = tsfel.time_series_features_extractor(cfg, nan_less_matrix)

print("spectral.shape", spectral.shape)
print("statistical.shape", statistical.shape)
print("temporal.shape", temporal.shape)
print("fractal.shape", fractal.shape)

# Iterate over each value in the row
#for col, value in X.iloc[0].items():  # Only one row, so use X.iloc[0]
#    print(f"{col}: {value}")


