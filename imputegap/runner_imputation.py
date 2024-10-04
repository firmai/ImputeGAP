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

    if algo == "cdrec":
        imp = Imputation.MD.CDRec(ts_2.data)
    elif algo == "stmvl":
        imp = Imputation.Pattern.STMVL(ts_2.data)
    elif algo == "iim":
        imp = Imputation.Regression.IIM(ts_2.data)
    elif algo == "mrnn":
        imp = Imputation.ML.MRNN(ts_2.data)
    else:
        imp = Imputation.Stats.MinImpute(ts_2.data)


    imp.optimize(ts_1.data, n_calls=2)
    imp.impute(imp.optimal_params)
    imp.score(ts_1.data)

    ts_3 = TimeSeries()
    ts_3.import_matrix(imp.imputed_matrix)
    ts_3.print(view_by_series=True)
    ts_3.print_results(imp.metrics, algorithm=algo)

    ts_3.plot(raw_matrix=ts_1.data, infected_matrix=ts_2.data, imputed_matrix=ts_3.data, title="imputation", max_series=1, save_path="assets", display=False)

    print("\n", "_"*95, "end")