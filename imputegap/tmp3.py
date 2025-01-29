import numpy as np
import pandas as pd  # Minimal Pandas usage for tsfresh
import tsfresh
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# Load time series data
ts_1 = TimeSeries()
ts_1.load_series(utils.search_path("chlorine"))

# Convert to NumPy array (Assumes shape = (num_series, num_values_per_series))
ts_data = ts_1.data  # Shape: (series, values_of_each_series)
num_series, num_values = ts_data.shape

# Create an index for each value in the time series
indices = np.tile(np.arange(num_values), num_series)  # Repeats 0,1,2,... for each series
series_ids = np.repeat(np.arange(num_series), num_values)  # Assigns unique ID to each series
values = ts_data.flatten()  # Convert 2D array to 1D for tsfresh


print("\nseries_ids", series_ids)
print("\nvalues", values)

# Convert to a minimal DataFrame for tsfresh processing
df = pd.DataFrame({"id": series_ids, "index": indices, "value": values})  # 'id' groups each series
# Extract features (M, 783)
features = tsfresh.extract_features(df, column_id="id", column_sort="index")
# Aggregate for the whole dataset (1, 783)
aggregated_features = features.mean(axis=0).to_frame().T


print("Features Shape for", num_series, " series :", features.shape)
print("Aggregated Features Shape:", aggregated_features.shape)

# Example of accessing the mean value of a specific feature
for feature, value in aggregated_features.iloc[0].items():
    print(f"{feature}: {value}")