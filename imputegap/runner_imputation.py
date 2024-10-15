from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

if __name__ == '__main__':
    dataset, algo = "eeg", "cdrec"
    utils.display_title()

    # 1. initiate the TimeSeries() object that will stay with you throughout the analysis
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_timeseries(utils.search_path(dataset), header=True)
    ts_1.normalize(normalizer="min_max")

    # [OPTIONAL] you can plot your raw data / print the information
    ts_1.plot(raw_data=ts_1.data, title="raw_data", max_series=10, save_path="assets", display=False)
    ts_1.print(limit=10)

    # 3. contamination of the data
    infected_data = ts_1.Contaminate.missing_percentage(ts_1.data)

    # [OPTIONAL] save your results in a new Time Series object
    ts_2 = TimeSeries().import_matrix(infected_data)

    # [OPTIONAL] you can plot your contaminated data / print the information
    ts_2.print(limit=5)
    ts_2.plot(raw_data=ts_1.data, infected_data=infected_data, title="contamination", max_series=1, save_path="assets", display=False)

    # 4. imputation of the contaminated data
    # choice of the algorithm, and their parameters (default, automl, or defined by the user)
    #cdrec = Imputation.MD.CDRec(infected_data).impute()
    cdrec = Imputation.MD.CDRec(infected_data).impute(user_defined=False, params={"ground_truth": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 2}})

    # [OPTIONAL] save your results in a new Time Series object
    ts_3 = TimeSeries().import_matrix(cdrec.imputed_matrix)

    # 5. score the imputation with the raw_data
    cdrec.score(ts_1.data, ts_3.data)

    # 6. display the results
    ts_3.print(view_by_series=True)
    ts_3.print_results(cdrec.metrics, algorithm=algo)
    ts_3.plot(raw_data=ts_1.data, infected_data=ts_2.data, imputed_data=ts_3.data, title="imputation", max_series=1, save_path="./assets", display=True)
