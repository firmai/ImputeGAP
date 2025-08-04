<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# Welcome to ImputeGAP

ImputeGAP is a comprehensive Python library for imputation of missing values in time series data. It implements user-friendly APIs to easily visualize, analyze, and repair time series datasets. The library supports a diverse range of imputation algorithms and modular missing data simulation catering to datasets with varying characteristics. ImputeGAP includes extensive customization options, such as automated hyperparameter tuning, benchmarking, explainability, and downstream evaluation.

In detail, the package provides:

  - Access to commonly used datasets in the time series imputation field ([Datasets](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets))
  - Configurable contamination that simulates real-world missingness patterns ([Patterns](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/recovery))
  - Parameterizable state-of-the-art time series imputation algorithms ([Algorithms](#Available-Imputation-Algorithms))
  - Extensive benchmarking to compare the performance of imputation algorithms ([Benchmark](#benchmark))
  - Modular tools to assess the impact of imputation on key downstream tasks ([Downstream](#downstream))
  - Fine-grained analysis of the impact of time series features on imputation results ([Explainer](#explainer))
  - Seamless integration of new algorithms in Python, C++, Matlab, Java, and R ([Contributing](https://imputegap.readthedocs.io/en/latest/contributing.html))

<br>

![Python](https://img.shields.io/badge/Python-v3.12-blue) ![Release](https://img.shields.io/badge/Release-v1.1.1-brightgreen)  ![License](https://img.shields.io/badge/License-GPLv3-blue?style=flat&logo=gnu) ![Coverage](https://img.shields.io/badge/Coverage-93%25-brightgreen) ![PyPI](https://img.shields.io/pypi/v/imputegap?label=PyPI&color=blue) ![Language](https://img.shields.io/badge/Language-English-blue) ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20MacOS-informational) [![Docs](https://img.shields.io/badge/Docs-available-brightgreen?style=flat&logo=readthedocs)](https://imputegap.readthedocs.io/)

<i>If you like our library, please add a ⭐ in our GitHub repository.</i>

<br>

| Tools                | URL                                                                                      |
|----------------------|------------------------------------------------------------------------------------------|
| 📚 **Documentation** | [https://imputegap.readthedocs.io/](https://imputegap.readthedocs.io/)                   |
| 📦 **PyPI**          | [https://pypi.org/project/imputegap/](https://pypi.org/project/imputegap/)               |
| 📁 **Datasets**      | [Description](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets) |


- ---


# Available Imputation Algorithms
| **Family**        | **Algorithm**             | **Venue -- Year**            |
|-------------------|---------------------------|------------------------------|
| LLMs              | NuwaTS [[35]](#ref35)     | Arxiv -- 2024                |
| LLMs              | GPT4TS [[36]](#ref36)     | NeurIPS -- 2023              |
| Deep Learning     | MissNet [[27]](#ref27)    | KDD -- 2024                  |
| Deep Learning     | MPIN [[25]](#ref25)       | PVLDB -- 2024                |
| Deep Learning     | BayOTIDE [[30]](#ref30)   | PMLR -- 2024                 |
| Deep Learning     | BitGraph [[32]](#ref32)   | ICLR -- 2024                 |
| Deep Learning     | PRISTI [[26]](#ref26)     | ICDE -- 2023                 |
| Deep Learning     | GRIN [[29]](#ref29)       | ICLR -- 2022                 |
| Deep Learning     | HKMF_T [[31]](#ref31)     | TKDE -- 2021                 |
| Deep Learning     | DeepMVI [[24]](#ref24)    | PVLDB -- 2021                |
| Deep Learning     | MRNN [[22]](#ref22)       | IEEE Trans on BE -- 2019     |
| Deep Learning     | BRITS [[23]](#ref23)      | NeurIPS -- 2018              |
| Deep Learning     | GAIN [[28]](#ref28)       | ICML -- 2018                 |
| Matrix Completion | CDRec [[1]](#ref1)        | KAIS -- 2020                 |
| Matrix Completion | TRMF [[8]](#ref8)         | NeurIPS -- 2016              |
| Matrix Completion | GROUSE [[3]](#ref3)       | PMLR -- 2016                 |
| Matrix Completion | ROSL [[4]](#ref4)         | CVPR -- 2014                 |
| Matrix Completion | SoftImpute [[6]](#ref6)   | JMLR -- 2010                 |
| Matrix Completion | SVT [[7]](#ref7)          | SIAM J. OPTIM -- 2010        |
| Matrix Completion | SPIRIT [[5]](#ref5)       | VLDB -- 2005                 |
| Matrix Completion | IterativeSVD [[2]](#ref2) | BIOINFORMATICS -- 2001       |
| Pattern Search    | TKCM [[11]](#ref11)       | EDBT -- 2017                 |
| Pattern Search    | STMVL [[9]](#ref9)        | IJCAI -- 2016                |
| Pattern Search    | DynaMMo [[10]](#ref10)    | KDD -- 2009                  |
| Machine Learning  | IIM [[12]](#ref12)        | ICDE -- 2019                 |
| Machine Learning  | XGBOOST [[13]](#ref13)    | KDD -- 2016                  |
| Machine Learning  | MICE [[14]](#ref14)       | Statistical Software -- 2011 |
| Machine Learning  | MissForest [[15]](#ref15) | BioInformatics -- 2011       |
| Statistics        | KNNImpute                 | -                            |
| Statistics        | Interpolation             | -                            |
| Statistics        | MinImpute                 | -                            |
| Statistics        | ZeroImpute                | -                            |
| Statistics        | MeanImpute                | -                            |
| Statistics        | MeanImputeBySeries        | -                            |

---

### **Quick Navigation**

- **Getting Started**  
  - [System Requirements](#system-requirements)  
  - [Installation](#installation)  

- **Code Snippets**  
  - [Dataset Loading](#dataset-loading)  
  - [Contamination](#contamination)  
  - [Imputation](#imputation)  
  - [Auto-ML](#Parameter-Tuning)  
  - [Explainer](#explainer)  
  - [Downstream Evaluation](#downstream)
  - [Benchmark](#benchmark)  
  - [Notebooks](#Jupyter-Notebooks)  

- **Contribute**  
  - [Integration Guide](#contribution)  

- **Additional Information**  
   - [Maintainers](#maintainers)  
   - [References](#references)  
 

---

<br> <br>

# Getting Started

## System Requirements

ImputeGAP runs with Python>=3.10 (except 3.13) and Unix-compatible environment.

<i>To create and set up an environment with Python 3.12, please refer to the [installation guide](https://imputegap.readthedocs.io/en/latest/getting_started.html).</i>


<br>

## Installation


### pip

To install/update the latest version of ImputeGAP, run the following command:

```bash
pip install imputegap
``` 

<br>

### Source

Alternatively, you can install the library from source:

```bash
git init
git clone https://github.com/eXascaleInfolab/ImputeGAP
cd ./ImputeGAP
pip install -e .
```

<br>

### Docker

Alternatively, you can download the latest version of ImputeGAP with all dependencies pre-installed using Docker.

Launch Docker and make sure it is running:

```bash
docker version
``` 

Pull the ImputeGAP Docker image (add `--platform linux/x86_64` in the command for MacOS) :

```bash
docker pull qnater/imputegap:1.1.2
```

Run the Docker container:

```bash
docker run -p 8888:8888 qnater/imputegap:1.1.2
``` 


---


<br> <br>

# Tutorials

## Dataset Loading

ImputeGAP comes with several time series datasets. The list of datasets is described [here](https://imputegap.readthedocs.io/en/latest/datasets.html).

As an example, we use the eeg-alcohol dataset, composed of individuals with a genetic predisposition to alcoholism. The dataset contains measurements from 64 electrodes placed on subject’s scalps, sampled at 256 Hz. The dimensions of the dataset are 64 series, each containing 256 values.


### Example Loading
You can find this example of normalization in the file [`runner_loading.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_loading.py).

To load and plot the eeg-alcohol dataset from the library:

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nImputeGAP datasets : {ts.datasets}")

# load and normalize the dataset from file or from the code
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# print and plot a subset of time series
ts.print(nbr_series=6, nbr_val=20)
ts.plot(input_data=ts.data, nbr_series=6, nbr_val=100, save_path="./imputegap_assets")
```

The module ``ts.datasets`` contains all the publicly available datasets provided by the library, which can be listed as follows:

```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"ImputeGAP datasets : {ts.datasets}")
```

---

## Contamination
We now describe how to simulate missing values in the loaded dataset. ImputeGAP implements eight different missingness patterns. For more details about the patterns, please refer to the documentation on this [page](https://imputegap.readthedocs.io/en/latest/patterns.html).
<br></br>

### Example Contamination
You can find this example in the file [`runner_contamination.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_contamination.py).


As example, we show how to contaminate the eeg-alcohol dataset with the MCAR pattern:

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# contaminate the time series with MCAR pattern
ts_m = ts.Contamination.mcar(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

# [OPTIONAL] plot the contaminated time series
ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets/contamination")
```

All missingness patterns developed in ImputeGAP are available in the ``ts.patterns`` module. They can be listed as follows:

```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"Missingness patterns : {ts.patterns}")
```



---

## Imputation

In this section, we will illustrate how to impute the contaminated time series. Our library implements five families of imputation algorithms: Statistical, Machine Learning, Matrix Completion, Deep Learning, and Pattern Search.
The list of algorithms is described [here](https://imputegap.readthedocs.io/en/latest/algorithms.html).

### Example Imputation
You can find this example in the file [`runner_imputation.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_imputation.py).

Let's illustrate the imputation using the CDRec algorithm from the Matrix Completion family.

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset
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
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")
```

Imputation can be performed using either default values or user-defined values. To specify the parameters, please use a dictionary in the following format:

```python
config = {"rank": 5, "epsilon": 0.01, "iterations": 100}
imputer.impute(params=config)
```

All algorithms developed in ImputeGAP are available in the ``ts.algorithms`` module, which can be listed as follows:
```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"Imputation families : {ts.families}")
print(f"Imputation algorithms : {ts.algorithms}")
```

---


## Parameter Tuning
The Optimizer component manages algorithm configuration and hyperparameter tuning. The parameters are defined by providing a dictionary containing the ground truth, the chosen optimizer, and the optimizer's options. Several search algorithms are available, including those provided by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).

### Example Auto-ML
You can find this example in the file [`runner_optimization.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_optimization.py).

Let's illustrate the imputation using the CDRec algorithm and Ray-Tune AutoML:

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# contaminate and impute the time series
ts_m = ts.Contamination.mcar(ts.data)
imputer = Imputation.MatrixCompletion.CDRec(ts_m)

# use Ray Tune to fine tune the imputation algorithm
imputer.impute(user_def=False, params={"input_data": ts.data, "optimizer": "ray_tune"})

# compute the imputation metrics with optimized parameter values
imputer.score(ts.data, imputer.recov_data)

# compute the imputation metrics with default parameter values
imputer_def = Imputation.MatrixCompletion.CDRec(ts_m).impute()
imputer_def.score(ts.data, imputer_def.recov_data)

# print the imputation metrics with default and optimized parameter values
ts.print_results(imputer_def.metrics, text="Default values")
ts.print_results(imputer.metrics, text="Optimized values")

# plot the recovered time series
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")

# save hyperparameters
utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg-alcohol", optimizer="ray_tune")
```
All optimizers developed in ImputeGAP are available in the ``ts.optimizers`` module, which can be listed as follows:

```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"AutoML Optimizers : {ts.optimizers}")
```


---



## Benchmark

ImputeGAP can serve as a common test-bed for comparing the effectiveness and efficiency of time series imputation algorithms[[33]](#ref33) . Users have full control over the benchmark by customizing various parameters,  including the list of the algorithms to compare, the optimizer, the datasets to evaluate, the missingness patterns, the range of missing values, and the performance metrics.


### Example Benchmark
You can find this example in the file [`runner_benchmark.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_benchmark.py).

The benchmarking module can be utilized as follows:

```python
from imputegap.recovery.benchmark import Benchmark

my_algorithms = ["SoftImpute", "MeanImpute"]

my_opt = ["default_params"]

my_datasets = ["eeg-alcohol"]

my_patterns = ["mcar"]

range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

my_metrics = ["*"]

# launch the evaluation
bench = Benchmark()
bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, metrics=my_metrics, optimizers=my_opt)
```

You can enable the optimizer using the following command:
```python
opt = {"optimizer": "ray_tune", "options": {"n_calls": 1, "max_concurrent_trials": 1}}
my_opt = [opt]
```


---


## Downstream
ImputeGAP includes a dedicated module for systematically evaluating the impact of data imputation on downstream tasks. Currently, forecasting is the primary supported task, with plans to expand to additional applications in the future.

### Example Downstream
You can find this example in the file [`runner_downstream.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_downstream.py).

Below is an example of how to call the downstream process for the model Prophet by defining a dictionary for the evaluator and selecting the model:

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset
ts.load_series(utils.search_path("forecast-economy"))
ts.normalize()

# contaminate the time series
ts_m = ts.Contamination.aligned(ts.data, rate_series=0.8)

# define and impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# compute and print the downstream results
downstream_config = {"task": "forecast", "model": "hw-add", "baseline": "ZeroImpute"}
imputer.score(ts.data, imputer.recov_data, downstream=downstream_config)
ts.print_results(imputer.downstream_metrics, text="Downstream results")
```

All downstream models developed in ImputeGAP are available in the ``ts.forecasting_models`` module, which can be listed as follows:

```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"ImputeGAP downstream models for forecasting : {ts.forecasting_models}")
```


---


## Explainer

The library provides insights into the algorithm’s behavior by identifying the features that impact the imputation results. It trains a regression model to predict imputation results across various methods and uses SHapley Additive exPlanations ([SHAP](https://shap.readthedocs.io/en/latest/)) to reveal how different time series features influence the model’s predictions.


### Example Explainer
You can find this example in the file [`runner_explainer.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_explainer.py).

Let’s illustrate the explainer using the CDRec algorithm and MCAR missingness pattern:


```python
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils

# initialize the time series and explainer object
ts = TimeSeries()
exp = Explainer()

# load and normalize the dataset
ts.load_series(utils.search_path("eeg-alcohol"))
ts.normalize(normalizer="z_score")

# configure the explanation
exp.shap_explainer(input_data=ts.data, extractor="pycatch", pattern="mcar", file_name=ts.name, algorithm="CDRec")

# print the impact of each feature
exp.print(exp.shap_values, exp.shap_details)

# plot the feature impacts
exp.show()
```

All feature extractors developed in ImputeGAP are available in the ``ts.extractors`` module, which can be listed as follows:

```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"ImputeGAP features extractors : {ts.extractors}")
```


---

## Jupyter Notebooks

ImputeGAP provides Jupyter notebooks available through the following links:

- [Jupyter: Imputegap Imputation Pipeline Notebook](https://github.com/eXascaleInfolab/ImputeGAP/blob/refs/heads/main/imputegap/notebooks/01_imputegap_pipeline_creation.ipynb)
- [Jupyter: Imputegap Advanced Analysis Notebook](https://github.com/eXascaleInfolab/ImputeGAP/blob/refs/heads/main/imputegap/notebooks/02_imputegap_advanced_analytics.ipynb)



## Google Colab Notebooks

ImputeGAP provides Google Colab notebooks available through the following links:

- [Google Colab: Imputegap Imputation Pipeline Notebook](https://colab.research.google.com/drive/1Kq1_HVoCTWLtB1zyryR35opxXmaprztV?usp=sharing)
- [Google Colab: Imputegap Advanced Analysis Notebook](https://colab.research.google.com/drive/1iOzLtpZTA3KDoyIc-srw2eoX5soEmP8x?usp=sharing)



---

## Contribution
To add your own imputation algorithm, please refer to the detailed [integration guide](https://imputegap.readthedocs.io/en/latest/contributing.html).


---


<br> <br>

# Citing

If you use ImputeGAP in your research, please cite these papers:

```
@article{nater2025imputegap,
  title = {ImputeGAP: A Comprehensive Library for Time Series Imputation},
  author = {Nater, Quentin and Khayati, Mourad and Pasquier, Jacques},
  year = {2025},
  eprint = {2503.15250},
  archiveprefix = {arXiv},
  primaryclass = {cs.LG},
  url = {https://arxiv.org/abs/2503.15250}
}
```

<br>

```
@article{nater2025kdd,
  title = {A Hands-on Tutorial on Time Series Imputation with ImputeGAP},
  author = {Nater, Quentin and Khayati, Mourad and Cudré-Mauroux, Philippe},
  year = {2025},
  booktitle = {SIGKDD Conference on Knowledge Discovery and Data Mining (To Appear)},
  series = {KDD2025}
}
```

<br> <br>

---

## Maintainers

<table>
  <tr>
    <td width="50%">
      <div  style="text-align: center;" >
        <a href="https://exascale.info/members/quentin-nater/">
            <img src="https://imputegap.readthedocs.io/en/latest/_images/quentin_nater.png" alt="Quentin Nater" width="150"/>
        </a><br>
        <strong>Quentin Nater</strong>
      </div>
      <br><br>
      <div style="text-align: justify;"> 
        Quentin is a PhD student jointly supervised by
        <a href="https://exascale.info/members/mourad-khayati/">Mourad Khayati</a> and
        <a href="https://exascale.info/phil/">Philippe Cudré-Mauroux</a> at the Department of Computer Science of the
        <a href="https://www.unifr.ch/home/en/">University of Fribourg</a>, Switzerland. He completed his Master’s degree in Digital Neuroscience at the University of Fribourg.
        His research focuses on <strong>time series analytics</strong>,
        including <strong>data imputation</strong>, <strong>machine learning</strong>, and
        <strong>multimodal learning</strong>.
      </div><br>
      👉 <a href="https://exascale.info/members/quentin-nater/">Home Page</a>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td width="50%" >
      <div  style="text-align: center;" >
        <a href="https://exascale.info/members/mourad-khayati/">
          <img src="https://imputegap.readthedocs.io/en/latest/_images/mourad_khayati.png" alt="Mourad Khayati" width="150" />
        </a><br>
        <strong>Mourad Khayati</strong>
      </div>
      <br><br>
      <div style="text-align: justify;"> 
        Mourad is a Senior Researcher and Lecturer with the
        <a href="https://exascale.info/">eXascale Infolab</a> and the Advanced Software Engineering group at the Department of Computer Science of the
        <a href="https://www.unifr.ch/home/en/">University of Fribourg</a>, Switzerland. 
        His research interests include <strong>time series analytics</strong> and
        <strong>data quality</strong>, with a focus on <strong>temporal data repair/cleaning</strong>.
        He received the <strong>VLDB 2020 Best Experiments and Analysis Paper Award</strong>.
      </div><br>
      👉 <a href="https://exascale.info/members/mourad-khayati/">Home Page</a>
    </td>
  </tr>
</table>


<br> <br>

---


## References

<a name="ref1"></a>
[1] Mourad Khayati, Philippe Cudré-Mauroux, Michael H. Böhlen: Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl. Inf. Syst. 62(6): 2257-2280 (2020)

<a name="ref2"></a>
[2] Olga G. Troyanskaya, Michael N. Cantor, Gavin Sherlock, Patrick O. Brown, Trevor Hastie, Robert Tibshirani, David Botstein, Russ B. Altman: Missing value estimation methods for DNA microarrays. Bioinform. 17(6): 520-525 (2001)

<a name="ref3"></a>
[3] Dejiao Zhang, Laura Balzano: Global Convergence of a Grassmannian Gradient Descent Algorithm for Subspace Estimation. AISTATS 2016: 1460-1468

<a name="ref4"></a>
[4] Xianbiao Shu, Fatih Porikli, Narendra Ahuja: Robust Orthonormal Subspace Learning: Efficient Recovery of Corrupted Low-Rank Matrices. CVPR 2014: 3874-3881

<a name="ref5"></a>
[5] Spiros Papadimitriou, Jimeng Sun, Christos Faloutsos: Streaming Pattern Discovery in Multiple Time-Series. VLDB 2005: 697-708

<a name="ref6"></a>
[6] Rahul Mazumder, Trevor Hastie, Robert Tibshirani: Spectral Regularization Algorithms for Learning Large Incomplete Matrices. J. Mach. Learn. Res. 11: 2287-2322 (2010)

<a name="ref7"></a>
[7] Jian-Feng Cai, Emmanuel J. Candès, Zuowei Shen: A Singular Value Thresholding Algorithm for Matrix Completion. SIAM J. Optim. 20(4): 1956-1982 (2010)

<a name="ref8"></a>
[8] Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon: Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction. NeurIPS 2016: 847-855

<a name="ref9"></a>
[9] Xiuwen Yi, Yu Zheng, Junbo Zhang, Tianrui Li: ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data. IJCAI 2016: 2704-2710

<a name="ref10"></a>
[10] Lei Li, James McCann, Nancy S. Pollard, Christos Faloutsos: DynaMMo: mining and summarization of coevolving sequences with missing values. 507-516

<a name="ref11"></a>
[11] Kevin Wellenzohn, Michael H. Böhlen, Anton Dignös, Johann Gamper, Hannes Mitterer: Continuous Imputation of Missing Values in Streams of Pattern-Determining Time Series. EDBT 2017: 330-341

<a name="ref12"></a>
[12] Aoqian Zhang, Shaoxu Song, Yu Sun, Jianmin Wang: Learning Individual Models for Imputation (Technical Report). CoRR abs/2004.03436 (2020)

<a name="ref13"></a>
[13] Tianqi Chen, Carlos Guestrin: XGBoost: A Scalable Tree Boosting System. KDD 2016: 785-794

<a name="ref14"></a>
[14] Royston Patrick , White Ian R.: Multiple Imputation by Chained Equations (MICE): Implementation in Stata. Journal of Statistical Software 2010: 45(4), 1–20.

<a name="ref15"></a>
[15] Daniel J. Stekhoven, Peter Bühlmann: MissForest - non-parametric missing value imputation for mixed-type data. Bioinform. 28(1): 112-118 (2012)

<a name="ref22"></a>
[22] Jinsung Yoon, William R. Zame, Mihaela van der Schaar: Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks. IEEE Trans. Biomed. Eng. 66(5): 1477-1490 (2019)

<a name="ref23"></a>
[23] Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, Yitan Li: BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018: 6776-6786

<a name="ref24"></a>
[24] Parikshit Bansal, Prathamesh Deshpande, Sunita Sarawagi: Missing Value Imputation on Multidimensional Time Series. Proc. VLDB Endow. 14(11): 2533-2545 (2021)

<a name="ref25"></a>
[25] Xiao Li, Huan Li, Hua Lu, Christian S. Jensen, Varun Pandey, Volker Markl: Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation (Extended Version). CoRR abs/2311.07344 (2023)

<a name="ref26"></a>
[26]: Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, Yanjie Fu: PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation. ICDE 2023: 1927-1939

<a name="ref27"></a>
[27] Kohei Obata, Koki Kawabata, Yasuko Matsubara, Yasushi Sakurai: Mining of Switching Sparse Networks for Missing Value Imputation in Multivariate Time Series. KDD 2024: 2296-2306

<a name="ref28"></a>
[28] Jinsung Yoon, James Jordon, Mihaela van der Schaar: GAIN: Missing Data Imputation using Generative Adversarial Nets. ICML 2018: 5675-5684

<a name="ref29"></a>
[29] Andrea Cini, Ivan Marisca, Cesare Alippi: Multivariate Time Series Imputation by Graph Neural Networks. CoRR abs/2108.00298 (2021)

<a name="ref30"></a>
[30] Shikai Fang, Qingsong Wen, Yingtao Luo, Shandian Zhe, Liang Sun: BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition. ICML 2024

<a name="ref31"></a>
[31] Liang Wang, Simeng Wu, Tianheng Wu, Xianping Tao, Jian Lu: HKMF-T: Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization. IEEE Trans. Knowl. Data Eng. 33(11): 3582-3593 (2021)

<a name="ref32"></a>
[32] Xiaodan Chen, Xiucheng Li, Bo Liu, Zhijun Li: Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values. ICLR 2024

<a name="ref33"></a>
[33] Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, Philippe Cudré-Mauroux: Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. Proc. VLDB Endow. 13(5): 768-782 (2020)

<a name="ref33"></a>
[34] Mourad Khayati, Quentin Nater, Jacques Pasquier: ImputeVIS: An Interactive Evaluator to Benchmark Imputation Techniques for Time Series Data. Proc. VLDB Endow. 17(12): 4329-4332 (2024)

<a name="ref35"></a>
[35] Jinguo Cheng, Chunwei Yang, Wanlin Cai, Yuxuan Liang, Qingsong Wen, Yuankai Wu: NuwaTS: a Foundation Model Mending Every Incomplete Time Series. Arxiv 2024

<a name="ref36"></a>
[36] Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin: One fits all: power general time series analysis by pretrained LM. NeurIPS 2023