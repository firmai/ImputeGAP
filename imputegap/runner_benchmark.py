from imputegap.recovery.benchmark import Benchmark

save_dir = "./imputegap_assets/benchmark"
nbr_runs = 1

datasets = ["eeg-alcohol"]

optimizers = ["default_params"]

algorithms = ["SoftImpute", "KNNImpute"]

patterns = ["mcar"]

range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

# launch the evaluation
list_results, sum_scores = Benchmark().eval(algorithms=algorithms, datasets=datasets, patterns=patterns, x_axis=range, optimizers=optimizers, save_dir=save_dir, runs=nbr_runs, verbose=False)