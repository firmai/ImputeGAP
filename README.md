![My Logo](assets/imputegab_logo.png)

# Welcome to ImputeGAP
ImputeGAP is a unified framework for imputation algorithms that provides a narrow-waist interface between algorithm evaluation and parameterization for datasets issued from various domains ranging from neuroscience, medicine, climate to energy.

The interface provides advanced imputation algorithms, construction of various missing values patterns, and different evaluation metrics. In addition, the framework offers support for AutoML parameterization techniques, feature extraction, and, potentially, analysis of feature impact using SHAP. The framework should allow a straightforward integration of new algorithms, datasets, and metrics.


## Installation
To install in local ImputeGAP, download the package from GitHub and run the command : 

```pip install -e .``` 

## Loading - Manager
The model of management is able to load any kind of time series datasets that respect this format : <b>(Values,Series), series are seperated by space et values by a carriage return \n.</b>

## Contamination
ImputeGAP allows to contaminate the dataset with a specific scenario to reproduce a situation. Up to now, the scenarios are : <b>MCAR, ...</b>