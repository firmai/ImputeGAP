from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils

from imputegap.recovery.manager import TimeSeries

ts_1 = TimeSeries()
ts_1.load_series(utils.search_path("chlorine"))

incomp_data = ts_1.Contamination.mcar(input_data=ts_1.data, dataset_rate=0.4, series_rate=0.4, block_size=10, offset=0.1, seed=True)

algo = Imputation.MatrixCompletion.CDRec(incomp_data)
algo.impute()
algo.score(ts_1.data)

_, metrics = algo.recov_data, algo.metrics

ts_1.print_results(algo.metrics, algorithm="cdrec")
