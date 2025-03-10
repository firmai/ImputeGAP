=========
Benchmark
=========

ImputeGAP offers a Benchmark module that enables users to compare various algorithm families across different dataset types using multiple evaluation metrics.

The number of runs determines the stability of results for Deep Learning algorithms, which may fluctuate during the imputation training process.
Users have full control over the analysis by customizing various parameters, including the list of datasets to evaluate, the choice of optimizer for fine-tuning algorithms on specific datasets, the algorithms to compare, the contamination patterns, and a range of missing rates.

The benchmarking module can be utilized as follows:

.. code-block:: python

    from imputegap.recovery.benchmark import Benchmark

    # VARIABLES
    save_dir = "./analysis"
    nbr_run = 2

    # SELECT YOUR DATASET(S) :
    datasets_demo = ["eeg-alcohol", "eeg-reading"]

    # SELECT YOUR OPTIMIZER :
    optimiser_bayesian = {"optimizer": "bayesian", "options": {"n_calls": 15, "n_random_starts": 50, "acq_func": "gp_hedge", "metrics": "RMSE"}}
    optimizers_demo = [optimiser_bayesian]

    # SELECT YOUR ALGORITHM(S) :
    algorithms_demo = ["mean", "cdrec", "stmvl", "iim", "mrnn"]

    # SELECT YOUR CONTAMINATION PATTERN(S) :
    patterns_demo = ["mcar"]

    # SELECT YOUR MISSING RATE(S) :
    x_axis = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    # START THE ANALYSIS
    list_results, sum_scores = Benchmark().eval(algorithms=algorithms_demo, datasets=datasets_demo, patterns=patterns_demo, x_axis=x_axis, optimizers=optimizers_demo, save_dir=save_dir, runs=nbr_run)





