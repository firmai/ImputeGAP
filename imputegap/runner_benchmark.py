from imputegap.tools import utils

from imputegap.recovery.benchmark import Benchmark


my_opt = ["default_params"]

my_datasets = ["meteo"]

my_patterns = ["mcar"]

range = [0.05, 0.8]

my_metrics = ["*"]

# launch the evaluation
bench = Benchmark()
bench.eval(algorithms=utils.list_of_algorithms(), datasets=my_datasets, patterns=my_patterns, x_axis=range, metrics=my_metrics, optimizers=my_opt, nbr_vals=500)