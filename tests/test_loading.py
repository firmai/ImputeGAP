import os
import unittest

from imputegap.manager._manager import TimeSeriesGAP


class TestLoading(unittest.TestCase):
    def test_loading_set(self):
        """
        Verify if the manager of a dataset is working
        """
        impute_gap = TimeSeriesGAP("./dataset/test.txt")

        self.assertEqual(impute_gap.ts.shape, (10, 25))
        self.assertEqual(not impute_gap.filename, False)
        self.assertEqual(impute_gap.ts[0, 1], 2.5)
        self.assertEqual(impute_gap.ts[1, 0], 0.5)

    def test_loading_chlorine(self):
        """
        Verify if the manager of a dataset is working
        """
        impute_gap = TimeSeriesGAP("./dataset/chlorine.txt")

        self.assertEqual(impute_gap.ts.shape, (50, 1000))
        self.assertEqual(not impute_gap.filename, False)
        self.assertEqual(impute_gap.ts[0, 1], 0.0154797)
        self.assertEqual(impute_gap.ts[1, 0], 0.0236836)

    def test_loading_plot(self):
        """
        Verify if the manager of a dataset is working
        """
        #if not hasattr(matplotlib.get_backend(), 'required_interactive_framework'):
        #    matplotlib.use('Agg')

        impute_gap = TimeSeriesGAP("./dataset/test.txt")
        filename = "./assets"

        impute_gap.plot("ground_truth", "test", filename, 5, (16, 8), False)

        self.assertTrue(os.path.exists(filename))