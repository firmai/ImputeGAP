from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.optimization import Optimization
from imputegap.tools import utils
from imputegap.tools.utils import display_title

def check_block_size(filename):
    if "test" in filename:
        return (2, 2)
    else:
        return (10, 10)


if __name__ == '__main__':

    display_title()
    dataset, algo = "eeg", "cdrec"


    ts_01 = TimeSeries()
    ts_01.load_timeseries(data=utils.search_path(dataset), max_series=50, max_values=100)

    block_size, plot_limit = check_block_size(dataset)

    infected_matrix = ts_01.Contaminate.mcar(ts=ts_01.data, series_impacted=0.4, missing_rate=0.4, block_size=block_size, protection=0.1, use_seed=True, seed=42)
    ts_01.print(limit=5)


    if algo == "cdrec":
        manager = Imputation.MD.CDRec(infected_matrix)
    elif algo == "mrnn":
        manager = Imputation.ML.MRNN(infected_matrix)
    elif algo == "iim":
        manager = Imputation.Regression.IIM(infected_matrix)
    else:
        manager = Imputation.Pattern.STMVL(infected_matrix)

    manager.optimize(raw_data=ts_01.data, optimizer="greedy", n_calls=25)
    manager.impute(params=("automl", ts_01.data, "greedy", 25))

    print("\nOptical Params : ", manager.parameters, "\n")

    Optimization.save_optimization(optimal_params=manager.parameters, algorithm=algo, dataset=dataset, optimizer="g")

    print("\n", "_" * 95, "end")