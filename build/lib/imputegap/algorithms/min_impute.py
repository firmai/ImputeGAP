import numpy as np


def min_impute(contamination, params=None):
    """
    Impute NaN values with the minimum value of the time series.

    Parameters
    ----------
    contamination : numpy.ndarray
        The input time series with contamination (missing values represented as NaNs).
    params : dict, optional
        Optional parameters for the algorithm. If None, the minimum value from the contamination is used (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix where NaN values have been replaced with the minimum value from the time series.

    Notes
    -----
    This function finds the minimum non-NaN value in the time series and replaces all NaN values with this minimum value.
    It is a simple imputation technique for filling missing data points in a dataset.

    Example
    -------
    >>> contamination = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    >>> imputed_matrix = min_impute(contamination)
    >>> print(imputed_matrix)
    array([[1., 2., 1.],
           [4., 1., 6.]])

    """

    # logic
    min_value = np.nanmin(contamination)

    # Imputation
    imputed_matrix = np.nan_to_num(contamination, nan=min_value)

    return imputed_matrix
