import numpy as np


def min_impute(contamination, params=None):
    """
    Impute NaN values with the minimum value of the ground truth time series.

    :param contamination: time series with contamination
    :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

    :return: imputed_matrix : all time series with imputation data
    """

    # logic
    min_value = np.nanmin(contamination)

    # Imputation
    imputed_matrix = np.nan_to_num(contamination, nan=min_value)

    return imputed_matrix
