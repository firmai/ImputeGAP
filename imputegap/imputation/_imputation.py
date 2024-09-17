import toml
from imputegap.algorithms.cdrec import native_cdrec_param
from imputegap.evaluation._evaluation import EvaluationGAP


class ImputationGAP:

    def __init__(self):
        """
        Initialize the ImputationGAP class.

        :param gap: the TimeSeriesGAP object managing the time series.
        """
        self.config = self.load_toml()


    def load_toml(self):
        """
        Load default values of algorithms
        :return: the config of default values
        """
        with open("../env/default_values.toml", "r") as file:
            config = toml.load(file)
        return config

    def metrics_computation(self, ground_truth, imputation, contamination):

        evaluation = EvaluationGAP(ground_truth, ground_truth+0.1, contamination) # test, to change

        rmse = evaluation.compute_rmse()
        mae = evaluation.compute_mae()
        mi_d = evaluation.compute_mi()
        correlation = evaluation.compute_correlation()

        return [rmse, mae, mi_d, correlation]

    def cdrec(self, ground_truth, contamination, params):
        """
        Imputation of data with CDREC algorithm
        @author Quentin Nater

        :param ground_truth: original time series without contamination
        :param contamination: time series with contamination
        :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

        :return: all time series with imputation data
        """

        if params is not None:
            truncation_rank, epsilon, iterations = params
        else:
            truncation_rank = self.config['cdrec']['default_reduction_rank']
            epsilon = self.config['cdrec']['default_epsilon_str']
            iterations = self.config['cdrec']['default_iteration']

        imputed_matrix = native_cdrec_param(__py_matrix=contamination, __py_rank=int(truncation_rank),
                                            __py_eps=float("1" + epsilon), __py_iters=int(iterations))


        metrics = self.metrics_computation(ground_truth, imputed_matrix, contamination)

        return imputed_matrix, metrics