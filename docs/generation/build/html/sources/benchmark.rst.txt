=========
Benchmark
=========

ImputeGAP can serve as a common test-bed for comparing the effectiveness and efficiency of time series imputation algorithms [33]_.  Users have full control over the benchmark by customizing various parameters, including the list of datasets to evaluate, the algorithms to compare, the choice of optimizer to fine-tune the algorithms on the chosen datasets, the missingness patterns, and the range of missing rates. The default metrics evaluated include "RMSE", "MAE", "MI", "Pearson", and the runtime.


The benchmarking module can be utilized as follows:

.. code-block:: python

    from imputegap.recovery.benchmark import Benchmark

    save_dir = "./imputegap_assets/benchmark"
    nbr_runs = 1

    datasets = ["eeg-alcohol"]

    optimizers = ["default_params"]

    algorithms = ["SoftImpute", "KNNImpute"]

    patterns = ["mcar"]

    range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    # launch the evaluation
    list_results, sum_scores = Benchmark().eval(algorithms=algorithms, datasets=datasets, patterns=patterns, x_axis=range, optimizers=optimizers, save_dir=save_dir, runs=nbr_runs)



.. [33] Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, Philippe Cudré-Mauroux: Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. Proc. VLDB Endow. 13(5): 768-782 (2020)





