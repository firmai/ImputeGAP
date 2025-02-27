<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# Welcome to ImputeGAP

ImputeGAP is a comprehensive framework designed for time series imputation algorithms. It offers a streamlined interface that bridges algorithm evaluation and parameter tuning, utilizing datasets from diverse fields such as neuroscience, medicine, and energy. The framework includes advanced imputation algorithms from five different families, supports various patterns of missing data, and provides multiple evaluation metrics. Additionally, ImputeGAP enables AutoML optimization, feature extraction, and feature analysis. The framework enables easy integration of new algorithms, datasets, and evaluation metrics.

![Python](https://img.shields.io/badge/Python-v3.12-blue) 
![Release](https://img.shields.io/badge/Release-v1.0.4-brightgreen) 
![License](https://img.shields.io/badge/License-GPLv3-blue?style=flat&logo=gnu)
![Coverage](https://img.shields.io/badge/Coverage-93%25-brightgreen)
![PyPI](https://img.shields.io/pypi/v/imputegap?label=PyPI&color=blue)
![Language](https://img.shields.io/badge/Language-English-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20MacOS-informational)
[![Docs](https://img.shields.io/badge/Docs-available-brightgreen?style=flat&logo=readthedocs)](https://exascaleinfolab.github.io/ImputeGAP/generation/build/html/index.html)

<br>

- **Documentation**: [https://exascaleinfolab.github.io/ImputeGAP/](https://exascaleinfolab.github.io/ImputeGAP/)
- **PyPI**: [https://pypi.org/project/imputegap/](https://pypi.org/project/imputegap/)
- **Datasets**: [Repository](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/dataset)
- ---

### **Quick Navigation**

- **Deployment**  
  - [System Requirements](#system-requirements)  
  - [Installation](#installation)  

- **Code Snippets**  
  - [Data Preprocessing](#loading-and-preprocessing)  
  - [Contamination](#contamination)  
  - [Imputation](#imputation)  
  - [Auto-ML](#parameterization)  
  - [Explainer](#explainer)  
  - [Downstream Evaluation](#downstream)
  - [Benchmark](#benchmark)  

- **Contribute**  
  - [Integration Guide](#integration)  

- **Additional Information**  
  - [References](#references)  
  - [Core Contributors](#core-contributors)  


---

## Category of Algorithms
| **CATEGORIES**    | **Algorithms**     | **CONTAMINATION** | **IMPUTATION** | **OPTIMIZATION** | **EXPLAINER** |
|:------------------|:-------------------|:--------------:|:-----------------:|:----------------:|:-------------:|
| Matrix Completion | CDRec              |       ✅        |         ✅         |        ✅         |       ✅       |
| Matrix Completion | IterativeSVD       |       ✅        |         ✅         |        ✅         |       ✅       |
| Matrix Completion | GROUSE             |       ✅        |         ✅         |        ✅         |       ✅       |
| Matrix Completion | ROSL               |       ✅        |         ✅         |        ✅         |       ✅       |
| Matrix Completion | SPIRIT             |       ✅        |         ✅         |        ✅         |       ✅       |
| Matrix Completion | SoftImpute         |       ✅        |         ✅         |        ✅         |       ✅       |
| Matrix Completion | SVT                |       ✅        |         ✅         |        ✅         |       ✅       |
| Matrix Completion | TRMF               |       ✅        |         ✅         |        ✅         |       ✅       |
| Pattern Search    | ST-MVL             |       ✅        |         ✅         |        ✅         |       ✅       |
| Pattern Search    | DynaMMo            |       ✅        |         ✅         |        ✅         |       ✅       |
| Pattern Search    | TKCM               |       ✅        |         ✅         |        ✅         |       ✅       |
| Machine Learning  | IIM                |       ✅        |         ✅         |        ✅         |       ✅       |
| Machine Learning  | XGBI               |       ✅        |         ✅         |        ✅         |       ✅       |
| Machine Learning  | Mice               |       ✅        |         ✅         |        ✅         |       ✅       |
| Machine Learning  | MissForest         |       ✅        |         ✅         |        ✅         |       ✅       |
| Statistics        | KNNImpute          |       ✅        |         ✅         |        ✅         |       ✅       |
| Statistics        | Interpolation      |       ✅        |         ✅         |        ✅         |       ✅       |
| Statistics        | MinImpute          |       ✅        |         ✅         |        ✅         |       ✅       |
| Statistics        | MeanImpute         |       ✅        |         ✅         |        ✅         |       ✅       |
| Statistics        | MeanImputeBySeries |       ✅        |         ✅         |        ✅         |       ✅       |
| Deep Learning     | MRNN               |       ✅        |         ✅         |        ✅         |       ✅       |
| Deep Learning     | BRITS              |       ✅        |         ✅         |        ✅         |       ✅       |
| Deep Learning     | DeepMVI            |       ✅        |         ✅         |        ✅         |       ✅       |
| Deep Learning     | MPIN               |       ✅        |         ✅         |        ✅         |       ✅       |
| Deep Learning     | PriSTI             |       ✅        |         ✅         |        ✅         |       ✅       |
| Deep Learning     | MissNet            |       ✅        |         ✅         |        ✅         |       ✅       |



---

## System Requirements

The following prerequisites are required to use ImputeGAP:

- Python version 3.10 / 3.11 / 3.12
- Unix-compatible environment for execution

To create and set up an environment with Python 3.12, please refer to the [installation guide](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/procedure/installation).


---


## Installation

### Pip installation

To quickly install the latest version of ImputeGAP along with its dependencies from the Python Package Index (PyPI), run the following command:

```bash
$ pip install imputegap
``` 


### Local installation

To modify the code of ImputeGAP or contribute to is development, you can install the library from source:

1) Initialize a Git repository and clone the project from GitHub:

```bash
$ git init
$ git clone https://github.com/eXascaleInfolab/ImputeGAP
$ cd ./ImputeGAP
``` 

2) Once inside the project directory, run the following command to install the package in editable mode:


```bash
$ pip install -e .
``` 

---
## Loading and Preprocessing

The data management module allows to load any time series datasets in text format, given they follow this
format: *(values, series)* with column separator: empty space, row separator: newline.

### Example Loading
You can find this example in the file [`runner_loading.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_loading.py).

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("eeg-alcohol"), max_series=5, max_values=15)
ts_1.normalize(normalizer="z_score")

# [OPTIONAL] you can plot your raw data / print the information
ts_1.plot(input_data=ts_1.data, max_series=10, max_values=100, save_path="./imputegap/assets")
ts_1.print(limit_series=10)
```

---

## Contamination
ImputeGAP allows to contaminate a complete datasets with missing data patterns that mimics real-world scenarios.<br></br>

The available patterns for MONO-BLOCK contamination : `MISSING POURCENTAGE`, `BLACKOUT`, `DISJOINT`, `OVERLAP` <br></br>
The available patterns for MULTI-BLOCK contamination : `MCAR`, `GAUSSIAN` and `DISTRIBUTION`.  <br></br>

For more details, please refer to the documentation in this <a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/recovery#readme" >page</a>.
<br></br>

### Example Contamination
You can find this example in the file [`runner_contamination.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_contamination.py).

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("eeg-alcohol"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data with MCAR pattern
ts_mask = ts_1.Contamination.mcar(ts_1.data, dataset_rate=0.2, series_rate=0.2, seed=True)

# [OPTIONAL] you can plot your raw data / print the contamination
ts_1.print(limit_timestamps=12, limit_series=7)
ts_1.plot(ts_1.data, ts_mask, max_series=9, subplot=True, save_path="./imputegap/assets")
```

---

## Imputation


ImputeGAP provides a diverse selection of imputation algorithms, organized into five main categories: Matrix Completion, Deep Learning, Statistical Methods, Pattern Search, and Machine Learning. You can also add your own custom imputation algorithm by following the `min-impute` template and substituting your code to implement your logic.

### Example Imputation
You can find this example in the file [`runner_imputation.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_imputation.py).

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("eeg-alcohol"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
ts_mask = ts_1.Contamination.mcar(ts_1.data)

# [OPTIONAL] save your results in a new Time Series object
ts_2 = TimeSeries().import_matrix(ts_mask)

# 4. imputation of the contaminated data
# choice of the algorithm, and their parameters (default, automl, or defined by the user)
imputer = Imputation.MatrixCompletion.CDRec(ts_2.data)

# imputation with default values
imputer.impute()
# OR imputation with user defined values
# >>> cdrec.impute(params={"rank": 5, "epsilon": 0.01, "iterations": 100})

# [OPTIONAL] save your results in a new Time Series object
ts_3 = TimeSeries().import_matrix(imputer.recov_data)

# 5. score the imputation with the raw_data
imputer.score(ts_1.data, ts_3.data)

# 6. display the results
ts_3.print_results(imputer.metrics, algorithm=imputer.algorithm)
ts_3.plot(input_data=ts_1.data, incomp_data=ts_2.data, recov_data=ts_3.data, max_series=9, subplot=True, save_path="./imputegap/assets")
```

---


## Parameterization
ImputeGAP provides optimization techniques that automatically identify the optimal hyperparameters for a specific algorithm in relation to a given dataset.
The available optimizers are: Greedy Optimizer (GO), Bayesian Optimizer (BO), Particle Swarm Optimizer (PSO), and Successive Halving (SH).

### Example Auto-ML
You can find this example in the file [`runner_optimization.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_optimization.py).

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("eeg-alcohol"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
ts_mask = ts_1.Contamination.mcar(ts_1.data)

# 4. imputation of the contaminated data
# imputation with AutoML which will discover the optimal hyperparameters for your dataset and your algorithm
imputer = Imputation.MatrixCompletion.CDRec(ts_mask).impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "ray_tune"})

# 5. score the imputation with the raw_data
imputer.score(ts_1.data, imputer.recov_data)

# 6. display the results
ts_1.print_results(imputer.metrics)
ts_1.plot(input_data=ts_1.data, incomp_data=ts_mask, recov_data=imputer.recov_data, max_series=9, subplot=True, save_path="./imputegap/assets", display=True)

# 7. save hyperparameters
utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg", optimizer="ray_tune")
```

---


## Explainer
ImputeGAP allows users to explore the features in the data that impact the imputation results
through Shapely Additive exPlanations ([**SHAP**](https://shap.readthedocs.io/en/latest/)). To attribute a meaningful interpretation of the SHAP results, ImputeGAP groups the extracted features into four categories: 
geometry, transformation, correlation, and trend.


### Example Explainer
You can find this example in the file [`runner_explainer.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_explainer.py).

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("eeg-alcohol"))

# 3. call the explanation of your dataset with a specific algorithm to gain insight on the Imputation results
shap_values, shap_details = Explainer.shap_explainer(input_data=ts_1.data, extractor="pycatch", pattern="mcar",
                                                     missing_rate=0.25, limit_ratio=1, split_ratio=0.7,
                                                     file_name="eeg-alcohol", algorithm="cdrec")

# [OPTIONAL] print the results with the impact of each feature.
Explainer.print(shap_values, shap_details)
```

---


## Downstream
ImputeGAP is a versatile library designed to help users evaluate both the upstream aspects (e.g., errors, entropy, correlation) and the downstream impacts of data imputation. By leveraging a built-in Forecaster, users can assess how the imputation process influences the performance of specific tasks.

### Example Downstream
You can find this example in the file [`runner_downstream.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_downstream.py).

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("chlorine"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
ts_mask = ts_1.Contamination.missing_percentage(ts_1.data, series_rate=0.8)
ts_2 = TimeSeries().import_matrix(ts_mask)

# 4. imputation of the contaminated data
imputer = Imputation.MatrixCompletion.CDRec(ts_2.data)
imputer.impute()

# [OPTIONAL] save your results in a new Time Series object
ts_3 = TimeSeries().import_matrix(imputer.recov_data)

# 5. score the imputation with the raw_data
downstream_options = {"evaluator": "forecaster", "model": "prophet"}
imputer.score(ts_1.data, ts_3.data)  # upstream standard analysis
imputer.score(ts_1.data, ts_3.data, downstream=downstream_options)  # downstream advanced analysis

# 6. display the results
ts_3.print_results(imputer.metrics, algorithm=imputer.algorithm)
ts_3.print_results(imputer.downstream_metrics, algorithm=imputer.algorithm)
```



---


## Benchmark
ImputeGAP enables users to comprehensively evaluate the efficiency of algorithms across various datasets.


### Example Benchmark
You can find this example in the file [`runner_benchmark.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_benchmark.py).

```python
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
```

---

## Integration
To add your own imputation algorithm in Python or C++, please refer to the detailed [integration guide](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/procedure/integration).


---


## References

Mourad Khayati, Quentin Nater, and Jacques Pasquier. ImputeVIS: An Interactive Evaluator to Benchmark Imputation Techniques for Time Series Data. Proceedings of the VLDB Endowment (PVLDB). Demo Track 17, no. 1 (2024), 4329 32.

Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, and Philippe Cudre-Mauroux. Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. In Proceedings of the VLDB Endowment (PVLDB), Vol. 13, 2020.


---


## Core Contributors
- Quentin Nater (<a href="mailto:quentin.nater@unifr.ch">quentin.nater@unifr.ch</a>)
- Dr. Mourad Khayati (<a href="mailto:mourad.khayati@unifr.ch">mourad.khayati@unifr.ch</a>)

