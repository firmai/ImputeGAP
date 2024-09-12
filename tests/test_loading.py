import os
import unittest

from imputegap.manager._manager import TimeSeriesGAP

commit = True

class TestLoading(unittest.TestCase):
    def test_loading_set(self):
        """
        Verify if the manager of a dataset is working
        """
        if commit:
            file_path = "./imputegap/dataset/test.txt"
        else:
            file_path = "./dataset/test.txt"

        impute_gap = TimeSeriesGAP(file_path)

        self.assertEqual(impute_gap.ts.shape, (10, 25))
        self.assertEqual(not impute_gap.filename, False)
        self.assertEqual(impute_gap.ts[0, 1], 2.5)
        self.assertEqual(impute_gap.ts[1, 0], 0.5)

    def test_loading_chlorine(self):
        """
        Verify if the manager of a dataset is working
        """
        if commit:
            file_path = "./imputegap/dataset/chlorine.txt"
        else:
            file_path = "./dataset/chlorine.txt"

        impute_gap = TimeSeriesGAP(file_path)

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

        if commit:
            file_path = "./imputegap/dataset/test.txt"
            save_path = "./imputegap/assets"
        else:
            file_path = "./dataset/test.txt"
            save_path = "./assets"

        impute_gap = TimeSeriesGAP(file_path)

        impute_gap.plot("ground_truth", "test", save_path, 5, (16, 8), False)

        self.assertTrue(os.path.exists(save_path))