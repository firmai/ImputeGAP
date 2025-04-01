from imputegap.recovery.benchmark import Benchmark

# create the ts object and load the time series
my_datasets = ["eeg-alcohol"]

my_opt = ["default_params"]

my_algorithms = ["SoftImpute", "KNNImpute"]

my_patterns = ["mcar"]

range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

# launch the evaluation
list_results, sum_scores = Benchmark().eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, metrics=["*"], optimizers=my_opt)