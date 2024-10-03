from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools.utils import display_title
from imputegap.tools import utils


if __name__ == '__main__':

    dataset, algo = "eeg", "cdrec"

    display_title()

    ts_1 = TimeSeries()
    ts_1.load_timeseries(utils.search_path(dataset))
    ts_1.plot(raw_matrix=ts_1.data, title="raw_data", max_series=1, save_path="assets", display=False)
    infected_matrix = ts_1.Contaminate.mcar(ts_1.data)

    ts_2 = TimeSeries()
    ts_2.import_matrix(infected_matrix)
    ts_2.print(view_by_series=True)
    ts_2.plot(raw_matrix=ts_1.data, infected_matrix=ts_2.data, title="contamination", max_series=1, save_path="assets", display=False)

    cdrec = Imputation.MD.CDREC(ts_2.data)
    cdrec.optimize(ts_1.data)
    cdrec.impute(cdrec.optimal_params)
    cdrec.score(ts_1.data, cdrec.imputed_matrix)

    ts_3 = TimeSeries()
    ts_3.import_matrix(cdrec.imputed_matrix)
    ts_3.print(view_by_series=True)
    ts_3.print_results(cdrec.metrics)

    ts_3.plot(raw_matrix=ts_1.data, infected_matrix=ts_2.data, imputed_matrix=ts_3.data, title="imputation", max_series=1, save_path="assets", display=False)

    print("\n", "_"*95, "end")