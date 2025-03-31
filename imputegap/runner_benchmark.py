from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

# create the ts object and load the time series
my_datasets = ["eeg-alcohol"]

my_opt = ["default_params"]

my_algorithms = ["GAIN", "MeanImpute"]
algs = ["CDRec",
        "IterativeSVD",
        "GROUSE",
        "ROSL",
        "SPIRIT",
        "SoftImpute",
        "SVT",
        "TRMF",
        "STMVL",
        "DynaMMo",
        "TKCM",
        "IIM",
        "XGBOOST",
        "MICE",
        "MissForest",
        "KNNImpute",
        "Interpolation",
        "MinImpute",
        "MeanImpute",
        "ZeroImpute",
        "MeanImputeBySeries",
        "MRNN",
        "BRITS",
        "DeepMVI",
        "PRISTI",
        "MissNet",
        "GAIN",
        "GRIN",
        "BayOTIDE",
        "HKMF_T",
        "BitGraph"
    ]
my_patterns = ["mcar"]

range = [0.2, 0.4, 0.6, 0.8]

# launch the evaluation
list_results, sum_scores = Benchmark().eval(algorithms=algs, datasets=my_datasets, patterns=my_patterns, x_axis=range, optimizers=my_opt)