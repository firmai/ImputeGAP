import numpy as np

from imputegap.wrapper.AlgoPython.IIM.testerIIM import impute_with_algorithm


def iim(contamination, number_neighbor, algo_code):
    """
    Template zero impute for adding your own algorithms
    @author : Quentin Nater

    :param contamination: time series with contamination
    :param adaptive_flag: The algorithm will run the non-adaptive version of the algorithm, as described in the paper
    :param number_neighbor : The number of neighbors to use for the KNN classifier, by default 10.
    :param algo_code : Action of the IIM output
    :return: imputed_matrix, metrics : all time series with imputation data and their metrics

    """
    #imputed_matrix = iim_recovery(matrix_nan=contamination, adaptive_flag=adaptive_flag, learning_neighbors=number_neighbor)
    imputed_matrix = impute_with_algorithm(algo_code, contamination.copy(), number_neighbor)

    return imputed_matrix
