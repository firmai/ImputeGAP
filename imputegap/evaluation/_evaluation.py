import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr




class EvaluationGAP:

    def __init__(self, ground_truth, imputation, contamination):
        """
        Initialize the EvaluationGAP class.

        :param ground_truth: original time series without contamination
        :param imputation: new time series with imputation values
        :param contamination: time series with contamination
        """
        self.ground_truth = ground_truth
        self.imputation = imputation
        self.contamination = contamination


    def compute_rmse(self):
        """
        Compute the RMSE score based on the ground_truth, the imputation values and the contamination set
        :return: the RMSE score between ground truth and imputation for NaN positions in contamination.
        """
        nan_locations = np.isnan(self.contamination)

        mse = np.mean((self.ground_truth[nan_locations] - self.imputation[nan_locations]) ** 2)
        rmse = np.sqrt(np.mean(mse))

        return float(rmse)


    def compute_mae(self):
        """
        Computes MAE only for the positions where there are NaN values in the contamination.
        :return : the mean absolute error between ground truth and imputation for NaN positions in contamination.
        """
        nan_locations = np.isnan(self.contamination)

        absolute_error = np.abs(self.ground_truth[nan_locations] - self.imputation[nan_locations])
        mean_absolute_error = np.mean(absolute_error)

        return mean_absolute_error

    def compute_mi(self):
        """
        Computes Mutual Information (MI) only for the positions where there are NaN values in the contamination.
        :return : the mutual information between ground truth and imputation for NaN positions in contamination.
        """
        nan_locations = np.isnan(self.contamination)

        # Discretize the continuous data into bins
        ground_truth_binned = np.digitize(self.ground_truth[nan_locations], bins=np.histogram_bin_edges(self.ground_truth[nan_locations], bins=10))
        imputation_binned = np.digitize(self.imputation[nan_locations], bins=np.histogram_bin_edges(self.imputation[nan_locations], bins=10))

        mi_discrete = mutual_info_score(ground_truth_binned, imputation_binned)
        #mi_continuous = mutual_info_score(self.ground_truth[nan_locations], self.ground_truth[nan_locations])

        return mi_discrete

    def compute_correlation(self):
        """
        Computes the Pearson correlation coefficient only for the positions where there are NaN values in the contamination.

        :return: the Pearson correlation coefficient between ground truth and imputation for NaN positions in contamination.
        """
        nan_locations = np.isnan(self.contamination)
        correlation, _ = pearsonr(self.ground_truth[nan_locations], self.imputation[nan_locations])

        return correlation