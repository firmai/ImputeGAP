from imputegap.wrapper.AlgoPython.MRNN.testerMRNN import mrnn_recov


def mrnn(contamination, hidden_dim, learning_rate, iterations, sequence_length):

    """
    Template zero impute for adding your own algorithms
    @author : Quentin Nater

    :param contamination: time series with contamination
    :param hidden_dim: number of hidden dimension
    :param learning_rate : learning rate of the training
    :param iterations : iterations during the training
    :param seq_length : length of the sequences inside MRNN
    :return: imputed_matrix, metrics : all time series with imputation data and their metrics

    """
    imputed_matrix = mrnn_recov(matrix_in=contamination, hidden_dim=hidden_dim, learning_rate=learning_rate, iterations=iterations, seq_length=sequence_length)

    return imputed_matrix
