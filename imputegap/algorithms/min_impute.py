import numpy as np


def min_impute(ground_truth, contamination, params=None):
    """
    Impute NaN values with the minimum value of the ground truth time series.

    :param ground_truth: original time series without contamination
    :param contamination: time series with contamination
    :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

    :return: imputed_matrix : all time series with imputation data
    """

    # logic
    min_value = np.nanmin(ground_truth)

    # Imputation
    imputed_matrix = np.nan_to_num(contamination, nan=min_value)

    return imputed_matrix
