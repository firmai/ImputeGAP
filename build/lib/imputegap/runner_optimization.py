from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.optimization import Optimization
from imputegap.tools import utils
from imputegap.tools.utils import display_title


if __name__ == '__main__':

    display_title()
    dataset, algorithm = "eeg", "cdrec"


    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_01 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_01.load_timeseries(data=utils.search_path(dataset), max_series=100, max_values=1000)

    # 3. contamination of the data
    infected_matrix = ts_01.Contaminate.mcar(ts=ts_01.data, use_seed=True, seed=42)

    # 4. config imputation depending on the algorithm
    if algorithm == "cdrec":
        manager = Imputation.MD.CDRec(infected_matrix)
    elif algorithm == "mrnn":
        manager = Imputation.ML.MRNN(infected_matrix)
    elif algorithm == "iim":
        manager = Imputation.Regression.IIM(infected_matrix)
    else:
        manager = Imputation.Pattern.STMVL(infected_matrix)

    # imputation with AutoML which will discover the optimal hyperparameters for your dataset and your algorithm
    manager.impute(user_defined=False, params={"ground_truth": ts_01.data, "optimizer": "bayesian", "options": {"n_calls": 5}})

    print("\nOptical Params : ", manager.parameters, "\n")

    # 5. score the imputation with the raw_data
    manager.score(ts_01.data)

    # 6. display results and save the optimal values in a persistent file
    ts_01.print_results(metrics=manager.metrics, algorithm=algorithm)
    Optimization.save_optimization(optimal_params=manager.parameters, algorithm=algorithm, dataset=dataset, optimizer="sh")
