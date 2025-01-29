import pandas as pd

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
import tsfel

ts_1 = TimeSeries()
ts_1.load_series(utils.search_path("chlorine"))


features_dic = {
        "1_LPCC_0": [0.1],
        "2_LPCC_0": [0.4],
        "3_MFCC_1": [0.7],
        "4_MFCC_1": [0.7],
        "555_steak": [0],
        "56_steak": [1],
    }
features_dic = pd.DataFrame.from_dict(features_dic)

#nan_less_matrix = incomp_data.copy()  # Copy to preserve the original matrix if needed
#nan_less_matrix[np.isnan(nan_less_matrix)] = 0
M, N = ts_1.data.shape

categories = ["spectral", "statistical", "temporal", "fractal"]
#spectral = features.filter(like="500", axis=1)

for category in categories:
    # the spectral configuration
    # Extract features with TSFEL
    cfg = tsfel.get_features_by_domain(category)
    features = tsfel.time_series_features_extractor(cfg, ts_1.data)

    # Extract feature types by removing the ID prefix
    features.columns = features.columns.str.split('_', n=1).str[1]

    # Group by feature type to handle duplicates and compute the mean
    aggregated_features = features.groupby(features.columns, axis=1).mean()

    # Convert to a single-row DataFrame (1x84)
    aggregated_features_df = pd.DataFrame([aggregated_features.mean(axis=0)])

    # Print the shape and aggregated features
    print(category, ": Aggregated features shape:", aggregated_features_df.shape)
    for col, value in aggregated_features_df.iloc[0].items():  # Only one row, so use X.iloc[0]
        print(f"{category}: {col}: {value}")
