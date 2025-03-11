<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# Welcome to ImputeGAP

ImputeGAP is a comprehensive framework designed for time series imputation algorithms. It offers a streamlined interface that bridges algorithm evaluation and parameter tuning, utilizing datasets from diverse fields such as neuroscience, medicine, and energy. The framework includes advanced imputation algorithms from five different families, supports various patterns of missing data, and provides multiple evaluation metrics. Additionally, ImputeGAP enables AutoML optimization, feature extraction, and feature analysis. The framework enables easy integration of new algorithms, datasets, and evaluation metrics.

![Python](https://img.shields.io/badge/Python-v3.12-blue) 
![Release](https://img.shields.io/badge/Release-v1.0.5-brightgreen) 
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

## Families of Algorithms
# Algorithms Table
| **Family**         | **Algorithm**               | **Venue -- Year**            |
|--------------------|-----------------------------|------------------------------|
| Matrix Completion  | CDRec [^1]                  | KAIS -- 2020                 |
| Matrix Completion  | IterativeSVD [^2]           | BIOINFORMATICS -- 2001       |
| Matrix Completion  | GROUSE [^3]                 | PMLR -- 2016                 |
| Matrix Completion  | ROSL [^4]                   | CVPR -- 2014                 |
| Matrix Completion  | SPIRIT [^5]                 | VLDB -- 2005                 |
| Matrix Completion  | SoftImpute [^6]             | JMLR -- 2010                 |
| Matrix Completion  | SVT [^7]                    | SIAM J. OPTIM -- 2010        |
| Matrix Completion  | TRMF [^8]                   | NeurIPS -- 2016              |
| Pattern Search     | ST-MVL [^9]                 | IJCAI -- 2016                |
| Pattern Search     | DynaMMo [^10]               | KDD -- 2009                  |
| Pattern Search     | TKCM [^11]                  | EDBT -- 2017                 |
| Machine Learning   | IIM [^12]                   | ICDE -- 2019                 |
| Machine Learning   | XGBI [^13]                  | KDD -- 2016                  |
| Machine Learning   | Mice [^14]                  | Statistical Software -- 2011 |
| Machine Learning   | MissForest [^15]            | BioInformatics -- 2011       |
| Statistics         | KNNImpute [^16]             | native                       |
| Statistics         | Interpolation [^17]         | native                       |
| Statistics         | Min Impute [^18]            | native                       |
| Statistics         | Zero Impute [^19]           | native                       |
| Statistics         | Mean Impute [^20]           | native                       |
| Statistics         | Mean Impute By Series [^21] | native                       |
| Deep Learning      | MRNN [^22]                  | IEEE Trans on BE -- 2019     |
| Deep Learning      | BRITS [^23]                 | NeurIPS -- 2018              |
| Deep Learning      | DeepMVI [^24]               | PVLDB -- 2021                |
| Deep Learning      | MPIN [^25]                  | PVLDB -- 2024                |
| Deep Learning      | PriSTI [^26]                | ICDE -- 2023                 |
| Deep Learning      | MissNet [^27]               | KDD -- 2024                  |
| Deep Learning      | GAIN [^28]                  | ICML -- 2018                 |
| Deep Learning      | GRIN [^29]                  | ICLR -- 2022                 |
| Deep Learning      | BayOTIDE [^30]              | PMLR -- 2024                 |
| Deep Learning      | HKMF-T [^31]                | TKDE -- 2021                 |
| Deep Learning      | BITGraph [^32]              | ICLR -- 2024                 |



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

ImputeGAP comes with several time series datasets. You can find them inside the submodule ``ts.datasets``.

As an example, we start by using eeg-alcohol, a standard dataset composed of individuals with a genetic predisposition to
alcoholism. The dataset contains measurements from 64 electrodes placed on subject’s scalps, sampled at 256 Hz (3.9-ms epoch) for 1 second. The dimensions of the dataset are 64 series, each containing 256 values.

### Example Loading
You can find this example in the file [`runner_loading.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_loading.py).

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initiate the TimeSeries() object that will stay with you throughout the analysis
ts = TimeSeries()

# load the timeseries from file or from the code
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# [OPTIONAL] plot a subset of time series
ts.plot(input_data=ts.data, nbr_series=9, nbr_val=100, save_path="./imputegap/assets")

# [OPTIONAL] print a subset of time series
ts.print(nbr_series=3, nbr_val=20)
```

---

## Contamination
We now describe how to simulate missing values in the loaded dataset. ImputeGAP implements eight different missingness patterns. You can find them inside the module ``ts.patterns``.


For more details, please refer to the documentation in this <a href="https://imputegap.readthedocs.io/en/latest/patterns.html" >page</a>.
<br></br>

### Example Contamination
You can find this example in the file [`runner_contamination.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_contamination.py).


As example, we show how to contaminate the eeg-alcohol dataset with the MCAR pattern:

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()

# load and normalize the timeseries
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# contamination of the data with MCAR pattern
ts_mask = ts.Contamination.mcar(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

# [OPTIONAL] plot the contaminated time series
ts.plot(ts.data, ts_mask, nbr_series=9, subplot=True, save_path="./imputegap/assets")
```

---

## Imputation

In this section, we will illustrate how to impute the contaminated time series. Our library implements five families of imputation algorithms. Statistical, Machine Learning, Matrix Completion, Deep Learning, and Pattern Search Methods.
You can find the list of algorithms inside the module ``ts.algorithms``.

```python
    params = {"param_1": value_1, "param_2": value_2, ...}
```

### Example Imputation
You can find this example in the file [`runner_imputation.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_imputation.py).

Imputation can be performed using either default values or user-defined values. To specify the parameters, please use a dictionary in the following format:

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()

# load and normalize the timeseries
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# contaminate the time series
ts_m = ts.Contamination.mcar(ts.data)

# impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# compute and print the imputation metrics
imputer.score(ts.data, imputer.recov_data)
ts.print_results(imputer.metrics)

# plot the recovered time series
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, save_path="./imputegap/assets")
```

---


## Parameterization
ImputeGAP provides optimization techniques that automatically identify the optimal hyperparameters for a specific algorithm in relation to a given dataset.
The available optimizers are: Greedy Optimizer (``greedy``), Bayesian Optimizer (``bo``), Particle Swarm Optimizer (``pso``), and Successive Halving (``sh``), Ray-Tune (``ray_tune``).

You can find the list of optimizers inside the module ``ts.optimizers``.

### Example Auto-ML
You can find this example in the file [`runner_optimization.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_optimization.py).

Let's illustrate the imputation using the CDRec Algorithm and Ray-Tune AutoML:

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()

# load and normalize the timeseries
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# contaminate the time series
ts_m = ts.Contamination.mcar(ts.data)

# define and impute the contaminated series with a optimizer
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute(user_def=False, params={"input_data": ts.data, "optimizer": "ray_tune"})

# compute and print the imputation metrics
imputer.score(ts.data, imputer.recov_data)
ts.print_results(imputer.metrics)

# plot the recovered time series
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, save_path="./imputegap/assets", display=True)

# save hyperparameters
utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg-alcohol", optimizer="ray_tune")
```

---


## Explainer
ImputeGAP allows users to explore the features in the data that impact the imputation results through Shapely Additive exPlanations ([**SHAP**](https://shap.readthedocs.io/en/latest/)).

To attribute a meaningful interpretation of the SHAP results, ImputeGAP groups the extracted features into four categories such as: geometry, transformation, correlation, and trend.


### Example Explainer
You can find this example in the file [`runner_explainer.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_explainer.py).

Let's illustrate the explainer using the CDRec Algorithm and MCAR missing pattern:


```python
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()

# load and normalize the timeseries
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# explanation of the imputation with a specific algorithm, pattern of contamination and dataset
shap_values, shap_details = Explainer.shap_explainer(input_data=ts.data, extractor="pycatch", pattern="mcar",
                                                     missing_rate=0.25, limit_ratio=1, split_ratio=0.7,
                                                     file_name="eeg-alcohol", algorithm="cdrec")

# print the impact of each feature
Explainer.print(shap_values, shap_details)
```

---


## Downstream
ImputeGAP is a versatile library designed to help users evaluate both the upstream aspects (e.g., errors, entropy, correlation) and the downstream impacts of data imputation.
By leveraging a built-in Forecaster, users can assess how the imputation process influences the performance of specific tasks.

### Example Downstream
You can find this example in the file [`runner_downstream.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_downstream.py).

Below is an example of how to call the downstream process for the model Prophet by defining a dictionary for the evaluator and selecting the model:

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the TimeSeries() object
ts = TimeSeries()

# load and normalize the timeseries
ts.load_series(utils.search_path("chlorine"))
ts.normalize(normalizer="min_max")

# contaminate the time series
ts_m = ts.Contamination.missing_percentage(ts.data, rate_series=0.8)

# define and impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# compute and print the imputation metrics with Up and Downstream
downstream_options = {"evaluator": "forecaster", "model": "prophet"}
imputer.score(ts.data, imputer.recov_data)  # upstream standard analysis
imputer.score(ts.data, imputer.recov_data, downstream=downstream_options)  # downstream advanced analysis

# print the imputation metrics
ts.print_results(imputer.metrics, algorithm=imputer.algorithm)
ts.print_results(imputer.downstream_metrics, algorithm=imputer.algorithm)


```



---


## Benchmark

ImputeGAP offers a Benchmark module that enables users to compare various algorithm families across different dataset types using multiple evaluation metrics.

The number of runs determines the stability of results for Deep Learning algorithms, which may fluctuate during the imputation training process.
Users have full control over the analysis by customizing various parameters, including the list of datasets to evaluate, the choice of optimizer for fine-tuning algorithms on specific datasets, the algorithms to compare, the contamination patterns, and a range of missing rates.

All missing data patterns developed in ImputeGAP are available in the ``ts.patterns`` module.
All algorithms developed in ImputeGAP are available in the ``ts.algorithms`` module.
All datasets provides in ImputeGAP are available in the ``ts.datasets`` module.
All optimizers developed in ImputeGAP are available in the ``ts.optimizers`` module.

### Example Benchmark
You can find this example in the file [`runner_benchmark.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_benchmark.py).

The benchmarking module can be utilized as follows:

```python
from imputegap.recovery.benchmark import Benchmark

# define analysis global variables
save_dir = "./analysis"
nbr_run = 2

# define the datasets to evaluate
datasets_demo = ["eeg-alcohol", "eeg-reading"]

# define the optimizer to fine-tine the algorithms
optimiser_bayesian = {"optimizer": "bayesian", "options": {"n_calls": 15, "n_random_starts": 50, "acq_func": "gp_hedge", "metrics": "RMSE"}}
optimizers_demo = [optimiser_bayesian]

# define the algorithms for the imputation
algorithms_demo = ["MeanImpute", "CDRec", "STMVL", "IIM", "MRNN"]

# define the missing pattern to contaminate the time series
patterns_demo = ["mcar"]

# define missing values percentages to see the evolution of the imputation
x_axis = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

# launch the analysis
list_results, sum_scores = Benchmark().eval(algorithms=algorithms_demo, datasets=datasets_demo, patterns=patterns_demo, x_axis=x_axis, optimizers=optimizers_demo, save_dir=save_dir, runs=nbr_run)
```

---

## Integration
To add your own imputation algorithm in Python or C++, please refer to the detailed [integration guide](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/procedure/integration).


---


## Articles

Mourad Khayati, Quentin Nater, and Jacques Pasquier. ImputeVIS: An Interactive Evaluator to Benchmark Imputation Techniques for Time Series Data. Proceedings of the VLDB Endowment (PVLDB). Demo Track 17, no. 1 (2024), 4329 32.

Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, and Philippe Cudre-Mauroux. Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. In Proceedings of the VLDB Endowment (PVLDB), Vol. 13, 2020.


---


## Core Contributors
- Quentin Nater (<a href="mailto:quentin.nater@unifr.ch">quentin.nater@unifr.ch</a>)
- Dr. Mourad Khayati (<a href="mailto:mourad.khayati@unifr.ch">mourad.khayati@unifr.ch</a>)


---


## References

[^1] Mourad Khayati, Philippe Cudré-Mauroux, Michael H. Böhlen: Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl. Inf. Syst. 62(6): 2257-2280 (2020)

[^2] Olga G. Troyanskaya, Michael N. Cantor, Gavin Sherlock, Patrick O. Brown, Trevor Hastie, Robert Tibshirani, David Botstein, Russ B. Altman: Missing value estimation methods for DNA microarrays. Bioinform. 17(6): 520-525 (2001)

[^3] Dejiao Zhang, Laura Balzano: Global Convergence of a Grassmannian Gradient Descent Algorithm for Subspace Estimation. AISTATS 2016: 1460-1468

[^4] Xianbiao Shu, Fatih Porikli, Narendra Ahuja: Robust Orthonormal Subspace Learning: Efficient Recovery of Corrupted Low-Rank Matrices. CVPR 2014: 3874-3881

[^5] Spiros Papadimitriou, Jimeng Sun, Christos Faloutsos: Streaming Pattern Discovery in Multiple Time-Series. VLDB 2005: 697-708

[^6] Rahul Mazumder, Trevor Hastie, Robert Tibshirani: Spectral Regularization Algorithms for Learning Large Incomplete Matrices. J. Mach. Learn. Res. 11: 2287-2322 (2010)

[^7] Jian-Feng Cai, Emmanuel J. Candès, Zuowei Shen: A Singular Value Thresholding Algorithm for Matrix Completion. SIAM J. Optim. 20(4): 1956-1982 (2010)

[^8] Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon: Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction. NIPS 2016: 847-855

[^9] Xiuwen Yi, Yu Zheng, Junbo Zhang, Tianrui Li: ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data. IJCAI 2016: 2704-2710

[^10] Lei Li, James McCann, Nancy S. Pollard, Christos Faloutsos: DynaMMo: mining and summarization of coevolving sequences with missing values. 507-516

[^11] Kevin Wellenzohn, Michael H. Böhlen, Anton Dignös, Johann Gamper, Hannes Mitterer: Continuous Imputation of Missing Values in Streams of Pattern-Determining Time Series. EDBT 2017: 330-341

[^12] Aoqian Zhang, Shaoxu Song, Yu Sun, Jianmin Wang: Learning Individual Models for Imputation (Technical Report). CoRR abs/2004.03436 (2020)

[^13] Tianqi Chen, Carlos Guestrin: XGBoost: A Scalable Tree Boosting System. KDD 2016: 785-794

[^14] Royston Patrick , White Ian R.: Multiple Imputation by Chained Equations (MICE): Implementation in Stata. Journal of Statistical Software 2010: 45(4), 1–20.

[^15] Daniel J. Stekhoven, Peter Bühlmann: MissForest - non-parametric missing value imputation for mixed-type data. Bioinform. 28(1): 112-118 (2012)

[^22] Jinsung Yoon, William R. Zame, Mihaela van der Schaar: Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks. IEEE Trans. Biomed. Eng. 66(5): 1477-1490 (2019)

[^23] Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, Yitan Li: BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018: 6776-6786

[^24] Parikshit Bansal, Prathamesh Deshpande, Sunita Sarawagi: Missing Value Imputation on Multidimensional Time Series. Proc. VLDB Endow. 14(11): 2533-2545 (2021)

[^25] Xiao Li, Huan Li, Hua Lu, Christian S. Jensen, Varun Pandey, Volker Markl: Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation (Extended Version). CoRR abs/2311.07344 (2023)

[^26] Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, Yanjie Fu: PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation. ICDE 2023: 1927-1939

[^27] Kohei Obata, Koki Kawabata, Yasuko Matsubara, Yasushi Sakurai: Mining of Switching Sparse Networks for Missing Value Imputation in Multivariate Time Series. KDD 2024: 2296-2306

[^28] Jinsung Yoon, James Jordon, Mihaela van der Schaar: GAIN: Missing Data Imputation using Generative Adversarial Nets. ICML 2018: 5675-5684

[^29] Andrea Cini, Ivan Marisca, Cesare Alippi: Multivariate Time Series Imputation by Graph Neural Networks. CoRR abs/2108.00298 (2021)

[^30] Shikai Fang, Qingsong Wen, Yingtao Luo, Shandian Zhe, Liang Sun: BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition. ICML 2024

[^31] Liang Wang, Simeng Wu, Tianheng Wu, Xianping Tao, Jian Lu: HKMF-T: Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization. IEEE Trans. Knowl. Data Eng. 33(11): 3582-3593 (2021)

[^32] Xiaodan Chen, Xiucheng Li, Bo Liu, Zhijun Li: Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values. ICLR 2024
