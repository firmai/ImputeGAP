import time
from imputegap.wrapper.AlgoPython.IIM.testerIIM import impute_with_algorithm


def iim(contamination, number_neighbor, algo_code, logs=True):
    """
    Template zero impute for adding your own algorithms
    :author: Quentin Nater

    :param contamination: time series with contamination
    :param adaptive_flag: The algorithm will run the non-adaptive version of the algorithm, as described in the paper
    :param number_neighbor : The number of neighbors to use for the KNN classifier, by default 10.
    :param algo_code : Action of the IIM output

    :param logs: print logs of time execution

    :return: imputed_matrix, metrics : all time series with imputation data and their metrics

    """
    start_time = time.time()  # Record start time

    imputed_matrix = impute_with_algorithm(algo_code, contamination.copy(), number_neighbor)

    end_time = time.time()
    if logs:
        print(f"\n\t\t> logs, imputation iim - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return imputed_matrix
