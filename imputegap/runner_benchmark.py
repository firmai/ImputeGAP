from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

my_algorithms = ["SoftImpute", "MeanImpute", "CDRec"]

my_opt = ["default_params"]

my_datasets = ["eeg-alcohol"]

my_patterns = ["mcar"]

range = [0.05, 0.1, 0.25, 0.4]

my_metrics = ["*"]

# launch the evaluation
bench = Benchmark()
bench.eval(algorithms=["BayOTIDE"], datasets=utils.list_of_datasets(), patterns=my_patterns, x_axis=range, metrics=my_metrics, optimizers=my_opt)