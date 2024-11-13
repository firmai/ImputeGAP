<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# Welcome to ImputeGAP

ImputeGAP is a comprehensive framework designed for imputation algorithms. It offers a streamlined interface that bridges algorithm evaluation and parameter tuning, utilizing datasets from diverse fields such as neuroscience, medicine, climate science, and energy.

The framework includes advanced imputation algorithms, supports various patterns of missing data, and provides multiple evaluation metrics. Additionally, ImputeGAP enables AutoML-based parameter optimization, feature extraction, and feature impact analysis with SHAP. The framework is built for easy integration of new algorithms, datasets, and evaluation metrics, enhancing its flexibility and adaptability.

![Python](https://img.shields.io/badge/Python-v3.12-blue) 
![Release](https://img.shields.io/badge/Release-v0.1.9-brightgreen) 
![License](https://img.shields.io/badge/License-GPLv3-blue?style=flat&logo=gnu)
![Coverage](https://img.shields.io/badge/Coverage-93%25-brightgreen)
![PyPI](https://img.shields.io/pypi/v/imputegap?label=PyPI&color=blue)
![Language](https://img.shields.io/badge/Language-English-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20MacOS-informational)
[![Docs](https://img.shields.io/badge/Docs-available-brightgreen?style=flat&logo=readthedocs)](https://exascaleinfolab.github.io/ImputeGAP/generation/build/html/index.html)

- **Documentation**: [https://exascaleinfolab.github.io/ImputeGAP/](https://exascaleinfolab.github.io/ImputeGAP/)
- **PyPI**: [https://pypi.org/project/imputegap/](https://pypi.org/project/imputegap/)
- **Datasets**: [Repository](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/dataset)

<br />

 [**Requirements**](#system-requirements) | [**Installation**](#installation) | [**Preprocessing**](#loading-and-preprocessing) | [**Contamination**](#contamination) | [**Auto-ML**](#parameterization) | [**Explainer**](#explainer) | [**Integration**](#integration) | [**Contributors**](#core-contributors)


---


## System Requirements

The following prerequisites are required to use ImputeGAP:

- Python version **3.12.0** to **3.12.6**
- Unix-compatible environment for execution

For instructions on installing these dependencies, please refer to the [installation guide](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/procedure/installation).



---

## Installation

### Python 3.12 installation

To install Python 3.12 on a Unix system, you can follow these steps before creating a virtual environment:

1) Update your package list and install prerequisites:

```
sudo apt-get update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev python3-tk libopenblas0 software-properties-common python3-pip
```

2) Add the deadsnakes PPA (for Ubuntu): This PPA provides newer Python versions for Ubuntu.

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
```

3) Install Python 3.12:

```
sudo apt-get install python3.12 python3.12-venv python3.12-dev
```

4) Verify the installation:
```
python3.12 --version
```

5) Create a virtual environment using Python 3.12:
```
python3.12 -m venv myenv
```

6) Activate the virtual environment:
```
source myenv/bin/activate
```

Now, you are ready to install your project or any dependencies within the Python 3.12 virtual environment.

<br />

### Pip installation

To quickly install the latest version of ImputeGAP along with its dependencies from the Python Package Index (PyPI), run the following command:

```bash
$ pip install imputegap
``` 

<br />

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
You can find this example in the file [`runner_contamination.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_contamination.py).

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("eeg-test"))
ts_1.normalize(normalizer="z_score")

# [OPTIONAL] you can plot your raw data / print the information
ts_1.plot(raw_data=ts_1.data, title="raw data", max_series=10, max_values=100, save_path="./imputegap/assets")
ts_1.print(limit=10)

```

---

## Contamination
ImputeGAP allows to contaminate a complete datasets with missing data patterns that mimics real-world scenarios. The available patterns are : <b>MCAR, MISSING POURCENTAGE, and BLACKOUT</b>. 
For more details, please refer to the documentation in this page : <a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/recovery#readme" >missing data patterns</a>.


### Example Contamination
You can find this example in the file [`runner_contamination.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_contamination.py).

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("eeg-test"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data with MCAR scenario
infected_data = ts_1.Contaminate.mcar(ts_1.data, series_impacted=0.4, missing_rate=0.2, use_seed=True)

# [OPTIONAL] you can plot your raw data / print the contamination
ts_1.print(limit=10)
ts_1.plot(ts_1.data, infected_data, title="contamination", max_series=1, save_path="./imputegap/assets")
```

---

## Imputation


ImputeGAP provides a diverse selection of imputation algorithms, organized into five main categories: Matrix Completion, Deep Learning, Statistical Methods, Pattern Search, and Graph Learning. You can also add your own custom imputation algorithm by following the `min-impute` template and substituting your code to implement your logic.

### Example Imputation
You can find this example in the file [`runner_imputation.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_imputation.py).

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("eeg-test"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
infected_data = ts_1.Contaminate.mcar(ts_1.data)

# 4. imputation of the contaminated data
# choice of the algorithm, and their parameters (default, automl, or defined by the user)
cdrec = Imputation.MatrixCompletion.CDRec(infected_data)

# imputation with default values
cdrec.impute()
# OR imputation with user defined values
cdrec.impute(params={"rank": 5, "epsilon": 0.01, "iterations": 100})

# [OPTIONAL] save your results in a new Time Series object
ts_3 = TimeSeries().import_matrix(cdrec.imputed_matrix)

# 5. score the imputation with the raw_data
cdrec.score(ts_1.data, ts_3.data)

# [OPTIONAL] print the results
ts_3.print_results(cdrec.metrics)
```


---
## Parameterization
ImputeGAP provides optimization techniques that automatically identify the optimal hyperparameters for a specific algorithm in relation to a given dataset.
The available optimizers are: Greedy Optimizer (GO), Bayesian Optimizer (BO), Particle Swarm Optimizer (PSO), and Successive Halving (SH.

### Example Auto-ML
You can find this example in the file [`runner_optimization.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_optimization.py).

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_timeseries(utils.search_path("eeg-test"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
infected_data = ts_1.Contaminate.mcar(ts_1.data)

# 4. imputation of the contaminated data
# imputation with AutoML which will discover the optimal hyperparameters for your dataset and your algorithm
cdrec = Imputation.MatrixCompletion.CDRec(infected_data).impute(user_defined=False, params={"ground_truth": ts_1.data,
                                                                                            "optimizer": "bayesian",
                                                                                            "options": {"n_calls": 5}})

# 5. score the imputation with the raw_data
cdrec.score(ts_1.data, cdrec.imputed_matrix)

# 6. [OPTIONAL] display the results
ts_1.print_results(cdrec.metrics)
ts_1.plot(raw_data=ts_1.data, infected_data=infected_data, imputed_data=cdrec.imputed_matrix, title="imputation",
          max_series=1, save_path="./assets", display=True)

# 7. [OPTIONAL] save hyperparameters
utils.save_optimization(optimal_params=cdrec.parameters, algorithm="cdrec", dataset="eeg", optimizer="b")
```


---

## Explainer
ImputeGAP allows users to explore the features in the data that impact the imputation results
through Shapely Additive exPlanations (SHAP). To attribute a meaningful interpretation of the SHAP results, ImputeGAP groups the extracted features into four categories: 
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
ts_1.load_timeseries(utils.search_path("eeg-test"))

# 3. call the explanation of your dataset with a specific algorithm to gain insight on the Imputation results
shap_values, shap_details = Explainer.shap_explainer(raw_data=ts_1.data, file_name="eeg-test", algorithm="cdrec")

# [OPTIONAL] print the results with the impact of each feature.
Explainer.print(shap_values, shap_details)
```

---

## Integration
To add your own imputation algorithm in Python or C++, please refer to the detailed [integration guide](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/procedure/integration).


---

## Core Contributors
- Quentin Nater (<a href="mailto:quentin.nater@unifr.ch">quentin.nater@unifr.ch</a>)
- Dr. Mourad Khayati (<a href="mailto:mourad.khayati@unifr.ch">mourad.khayati@unifr.ch</a>)

