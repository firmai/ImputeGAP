import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

from imputegap.tools import utils

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error

class Downstream:
    """
    A class to evaluate the performance of imputation algorithms using downstream analysis.

    This class provides tools to assess the quality of imputed time series data by analyzing
    the performance of downstream forecasting models. It computes metrics such as Mean Absolute
    Error (MAE) and Mean Squared Error (MSE) and visualizes the results for better interpretability.

    Attributes
    ----------
    input_data : numpy.ndarray
        The original time series without contamination (ground truth).
    recov_data : numpy.ndarray
        The imputed time series to evaluate.
    incomp_data : numpy.ndarray
        The time series with contamination (NaN values).
    downstream : dict
        Configuration for the downstream analysis, including the evaluator, model, and parameters.
    split : float
        The proportion of data used for training in the forecasting task (default is 0.8).

    Methods
    -------
    __init__(input_data, recov_data, incomp_data, downstream)
        Initializes the Downstream class with the provided data and configuration.
    downstream_analysis()
        Performs downstream analysis, computes metrics, and optionally visualizes results.
    _plot_downstream(y_train, y_test, y_pred, incomp_data, title="Ground Truth vs Predictions", max_series=4)
        Static method to plot ground truth vs. predictions for contaminated series.
    """



    def __init__(self, input_data, recov_data, incomp_data, downstream):
        """
        Initialize the Downstream class

        Parameters
        ----------
        input_data : numpy.ndarray
            The original time series without contamination.
        recov_data : numpy.ndarray
            The imputed time series.
        incomp_data : numpy.ndarray
            The time series with contamination (NaN values).
        downstream : dict
            Information about the model to launch with its parameters
        """
        self.input_data = input_data
        self.recov_data = recov_data
        self.incomp_data = incomp_data
        self.downstream = downstream
        self.split = 0.8

    def downstream_analysis(self):
        """
        Compute a set of evaluation metrics with a downstream analysis

        Returns
        -------
        dict or None
            Metrics from the downstream analysis or None if no valid evaluator is provided.
        """
        evaluator = self.downstream.get("task", "forecast")
        model = self.downstream.get("model", "naive")
        params = self.downstream.get("params", None)
        plots = self.downstream.get("plots", True)

        if not params:
            print("\n\t\t\t\tThe params for model of downstream analysis are empty or missing. Default ones loaded...")
            loader = "forecaster-" + str(model)
            params = utils.load_parameters(query="default", algorithm=loader)

        print("\n\t\t\t\tDownstream analysis launched for <", evaluator, "> on the model <", model,
              "> with parameters :\n\t\t\t\t\t", params)

        if evaluator == "forecast" or evaluator == "forecaster"or evaluator == "forecasting":
            y_train_all, y_test_all, y_pred_all = [], [], []
            mae, mse = [], []

            for x in range(3):  # Iterate over recov_data, input_data, and mean_impute
                if x == 0:
                    data = self.recov_data
                elif x == 1:
                    data = self.input_data
                elif x == 2:
                    from imputegap.recovery.imputation import Imputation
                    zero_impute = Imputation.Statistics.ZeroImpute(self.incomp_data).impute()
                    data = zero_impute.recov_data

                data_len = data.shape[1]
                train_len = int(data_len * self.split)

                y_train = data[:, :train_len]
                y_test = data[:, train_len:]
                y_pred = np.zeros_like(y_test)

                # Forecast for each series
                for series_idx in range(data.shape[0]):
                    series_train = y_train[series_idx, :]

                    # Initialize and fit the forecasting model
                    if model == "prophet":
                        forecaster = Prophet(**params)
                    elif model == "exp-smoothing":
                        forecaster = ExponentialSmoothing(**params)
                    else:
                        forecaster = NaiveForecaster(**params)

                    forecaster.fit(series_train)
                    fh = np.arange(1, y_test.shape[1] + 1)  # Forecast horizon
                    series_pred = forecaster.predict(fh=fh)
                    series_pred = series_pred.ravel()

                    # Store predictions
                    y_pred[series_idx, :] = series_pred

                # Validate shapes
                if y_pred.shape != y_test.shape:
                    raise ValueError(f"Shape mismatch: y_pred={y_pred.shape}, y_test={y_test.shape}, y_train={y_train.shape}")

                # Calculate metrics
                mae.append(mean_absolute_error(y_test, y_pred))
                mse.append(mean_squared_error(y_test, y_pred))

                # Store for plotting
                y_train_all.append(y_train)
                y_test_all.append(y_test)
                y_pred_all.append(y_pred)

            if plots:
                # Global plot with all rows and columns
                self._plot_downstream(y_train_all, y_test_all, y_pred_all, self.incomp_data)

            # Save metrics in a dictionary
            metrics = {"DOWNSTREAM-RECOV-MAE": mae[0], "DOWNSTREAM-INPUT-MAE": mae[1],
                       "DOWNSTREAM-MEANI-MAE": mae[2], "DOWNSTREAM-RECOV-MSE": mse[0],
                       "DOWNSTREAM-INPUT-MSE": mse[1], "DOWNSTREAM-MEANI-MSE": mse[2]}

            print("\t\t\t\tDownstream analysis complete. " + "*" * 58 + "\n")

            return metrics
        else:
            print("\t\t\t\tNo evaluator found... list possible : 'forecaster'" + "*" * 30 + "\n")

            return None

    @staticmethod
    def _plot_downstream(y_train, y_test, y_pred, incomp_data, title="Ground Truth vs Predictions", max_series=4, save_path="./imputegap/assets"):
        """
        Plot ground truth vs. predictions for contaminated series (series with NaN values).

        Parameters
        ----------
        y_train : np.ndarray
            Training data array of shape (n_series, train_len).
        y_test : np.ndarray
            Testing data array of shape (n_series, test_len).
        y_pred : np.ndarray
            Forecasted data array of shape (n_series, test_len).
        incomp_data : np.ndarray
            Incomplete data array of shape (n_series, total_len), used to identify contaminated series.
        title : str
            Title of the plot.
        max_series : int
            Maximum number of series to plot (default is 9).
        """
        # Create a 3x3 subplot grid (3 rows for data types, 3 columns for valid series)

        x_size = max_series * 5

        fig, axs = plt.subplots(3, max_series, figsize=(x_size, 15))
        fig.suptitle(title, fontsize=16)

        # Iterate over the three data types (recov_data, input_data, mean_impute)
        for row_idx in range(len(y_train)):
            # Find indices of the first 4 valid (non-NaN) series
            valid_indices = [i for i in range(incomp_data.shape[0]) if np.isnan(incomp_data[i]).any()][:max_series]

            for col_idx, series_idx in enumerate(valid_indices):
                # Access the correct subplot
                ax = axs[row_idx, col_idx]

                # Extract the corresponding data for this data type and series
                s_y_train = y_train[row_idx]
                s_y_test = y_test[row_idx]
                s_y_pred = y_pred[row_idx]

                # Combine training and testing data for visualization
                full_series = np.concatenate([s_y_train[series_idx], s_y_test[series_idx]])

                # Plot training data
                ax.plot(range(len(s_y_train[series_idx])), s_y_train[series_idx], label="Training Data", color="green")

                # Plot ground truth (testing data)
                ax.plot(
                    range(len(s_y_train[series_idx]), len(full_series)),
                    s_y_test[series_idx],
                    label="Ground Truth",
                    color="green"
                )

                # Plot forecasted data
                ax.plot(
                    range(len(s_y_train[series_idx]), len(full_series)),
                    s_y_pred[series_idx],
                    label="Forecast",
                    linestyle="--",
                    marker=None,
                    color="red"
                )

                # Add a vertical line at the split point
                ax.axvline(x=len(s_y_train[series_idx]), color="orange", linestyle="--", label="Split Point")

                # Add labels, title, and grid
                if row_idx == 0:
                    ax.set_title(f"IMPUTATED DATA (RECOVERY), series {series_idx}")
                elif row_idx == 1:
                    ax.set_title(f"ORIGINAL DATA (GROUND TRUTH), series {series_idx}")
                else:
                    ax.set_title(f"BAD IMPUTER (ZERO IMP), series {series_idx}")

                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid()

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            now = datetime.datetime.now()
            current_time = now.strftime("%y_%m_%d_%H_%M_%S")
            file_path = os.path.join(save_path + "/" + current_time + "_downstream.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            print("plots saved in ", file_path)

        plt.show()
