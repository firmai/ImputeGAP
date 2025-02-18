import numpy as np

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()
ts_1.load_series(utils.search_path("airq"))
ts_1.data = ts_1.data.T

ms_data = TimeSeries()
ms_data.load_series(utils.search_path("airq_m200"), replace_nan=False)
ms_data.data = ms_data.data.T

scores = []
for x in range(0, 3):
    algo = Imputation.DeepLearning.MPIN(ms_data.data).impute()

    print("ts_1.data", ts_1.data.shape)
    print("algo.incomp_data", algo.incomp_data.shape)
    print("algo.recov_data", algo.recov_data.shape)

    algo.score(ts_1.data, algo.recov_data)

    scores.append(algo.metrics["RMSE"])

    metrics = algo.metrics
    ts_1.print_results(algo.metrics, algorithm=algo.algorithm)

scores = np.array(scores)

avg = np.mean(scores)

print("\n\tAVG", avg)
