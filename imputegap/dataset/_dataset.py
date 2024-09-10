import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


class TimeSeriesLoader:

    def __init__(self, filename):
        self.filename = filename
        self.data = None

    def load_timeseries (self) :
        """
        Load timeseries dataset from file (ROW : Values/COL : Series)
        @author Quentin Nater

        :param filename: path of the file
        :return: panda set of series transposed
        """

        print("\t\t >> LOAD SERIES " + str(self.filename))
        sets = pd.read_csv(self.filename, delim_whitespace=True, header=None)
        sets = sets.transpose()

        if "electricity" in self.filename:
            sets = sets.transpose()

        self.data = sets

        return self.data