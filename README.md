![My Logo](https://www.naterscreations.com/imputegap/logo_imputegab.png)

# Welcome to ImputeGAP

ImputeGAP is a comprehensive framework designed for imputation algorithms. It offers a streamlined interface that bridges algorithm evaluation and parameter tuning, utilizing datasets from diverse fields such as neuroscience, medicine, climate science, and energy.

The framework includes advanced imputation algorithms, supports various patterns of missing data, and provides multiple evaluation metrics. Additionally, ImputeGAP enables AutoML-based parameter optimization, feature extraction, and feature impact analysis with SHAP. The framework is built for easy integration of new algorithms, datasets, and evaluation metrics, enhancing its flexibility and adaptability.


- **Documentation**: [https://exascaleinfolab.github.io/ImputeGAP/](https://exascaleinfolab.github.io/ImputeGAP/)
- **PyPI**: [https://pypi.org/project/imputegap/](https://pypi.org/project/imputegap/)



<br /><hr /><br />



## System Requirements

To use **ImputeGAP**, the following prerequisites are necessary:

- Python version **3.12.0** or higher
- A **Unix-compatible environment** for execution

For instructions on installing these dependencies, please refer to the [installation guide](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/procedure/installation).



<br /><hr /><br />




## Installation



### Pip installation

To quickly install the latest version of ImputeGAP along with its dependencies from the Python Package Index (PyPI), run the following command:

```bash
$ pip install imputegap
``` 


<br />


### Local installation
To install ImputeGAP from source, follow these steps:

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

This installation method is recommended if you want to modify the source code or contribute to the development of ImputeGAP.

<br /><hr /><br />

## Datasets
All preconfigured datasets available in this library can be accessed at the following location: [Link to datasets](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/dataset)


<br /><hr /><br />



## Loading and Pre-processing

The data management module allows to load any time series datasets in text format, given they follow this format: *(values, series)* with column separator: empty space, row separator: newline

To check if your dataset is correctly formatted, please refer to the example datasets provided [above](#datasets).<br />


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

<br /><hr /><br />



## Contamination
ImputeGAP allows to contaminate datasets with a specific scenario to reproduce a real-world situation. Up to now, the scenarios are : <b>MCAR, MISSING POURCENTAGE, and BLACKOUT</b><br />
Please find the documentation in this page : <a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/recovery#readme" >missing data scenarios</a><br><br>


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

<br /><hr /><br />

## Imputation
**ImputeGAP** offers a wide range of imputation algorithms categorized into several families, including: **Matrix Completion**, **Deep Learning**, **Statistics**, **Pattern Search**, and **Graphs Learning**.

It is also possible to add your own custom imputation algorithm. To do this, simply follow the `min-impute` template and replace the logic with your own code.

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


<br /><hr /><br />

## Auto-ML
**ImputeGAP** provides optimization techniques that automatically identify the optimal hyperparameters for a specific algorithm in relation to a given dataset.

The available optimizers include: **Greedy Optimizer**, **Bayesian Optimizer**, **Particle Swarm Optimizer**, and **Successive Halving**.

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


<br /><hr /><br />


## Explainer
**ImputeGAP** includes an algorithm based on the **SHAP** library, which explains the results of your imputations using features specific to your dataset.

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


<br /><hr /><br />

## Contributors
Quentin Nater (<a href="mailto:quentin.nater@unifr.ch">quentin.nater@unifr.ch</a>) and Dr. Mourad Khayati (<a href="mailto:mourad.khayati@unifr.ch">mourad.khayati@unifr.ch</a>)

<br /><br />
