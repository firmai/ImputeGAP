import numpy as np


def zero_impute(ground_truth, contamination, params=None):
    """
    Template zero impute for adding your own algorithms
    @author : Quentin Nater

    :param ground_truth: original time series without contamination
    :param contamination: time series with contamination
    :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

    :return: imputed_matrix : all time series with imputation data
    """
    imputed_matrix = np.nan_to_num(contamination, nan=0)

    return imputed_matrix
