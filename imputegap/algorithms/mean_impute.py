import numpy as np


def mean_impute(contamination, params=None):
    """
    Impute NaN values with the mean value of the time series.

    Parameters
    ----------
    contamination : numpy.ndarray
        The input time series with contamination (missing values represented as NaNs).
    params : dict, optional
        Optional parameters for the algorithm. If None, the minimum value from the contamination is used (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix where NaN values have been replaced with the mean value from the time series.

    Notes
    -----
    This function finds the non-NaN value in the time series and replaces all NaN values with this mean value.
    It is a simple imputation technique for filling missing data points in a dataset.

    Example
    -------
    >>> contamination = np.array([[5, 2, np.nan], [3, np.nan, 6]])
    >>> imputed_matrix = mean_impute(contamination)
    >>> print(imputed_matrix)
    array([[5., 2., 4.],
           [3., 4., 6.]])

    """

    # logic
    mean_value = np.nanmean(contamination)

    # Imputation
    imputed_matrix = np.nan_to_num(contamination, nan=mean_value)

    return imputed_matrix
