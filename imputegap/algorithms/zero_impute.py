import numpy as np


def zero_impute(ground_truth, contamination, params=None):
    """
    Template zero impute for adding your own algorithms
    :param imputegap:
    :return: imputation matrix
    """
    imputation = np.nan_to_num(contamination, nan=0)

    return imputation
