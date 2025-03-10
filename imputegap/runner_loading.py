from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()


print("ts_1.algorithms", ts_1.algorithms, "\n")
print("ts_1.datasets", ts_1.datasets, "\n")
print("ts_1.datasets 2 ", utils.list_of_datasets(txt=True), "\n")
print("ts_1.patterns", ts_1.patterns, "\n")
print("ts_1.optimizers", ts_1.optimizers, "\n")


# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("test"))
ts_1.normalize(normalizer="z_score")

# [OPTIONAL] you can plot your raw data / print the information
ts_1.plot(input_data=ts_1.data, save_path="./imputegap/assets")
ts_1.print()

