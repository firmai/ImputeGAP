imputegap documentation
=======================

ImputeGAP is a unified framework for imputation algorithms that provides a narrow-waist interface between algorithm evaluation and parameterization for datasets issued from various domains ranging from neuroscience, medicine, climate to energy.

The interface provides advanced imputation algorithms, construction of various missing values patterns, and different evaluation metrics. In addition, the framework offers support for AutoML parameterization techniques, feature extraction, and, potentially, analysis of feature impact using SHAP. The framework should allow a straightforward integration of new algorithms, datasets, and metrics.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

.. autosummary::
   :toctree: _autosummary

   imputegap.recovery.manager
   imputegap.recovery.imputation
   imputegap.recovery.optimization
   imputegap.recovery.explainer
   imputegap.recovery.evaluation
   imputegap.recovery.benchmarking
   imputegap.algorithms.cdrec
   imputegap.algorithms.stmvl
   imputegap.algorithms.iim
   imputegap.algorithms.mrnn
   imputegap.algorithms.mean_impute
   imputegap.algorithms.min_impute
   imputegap.algorithms.zero_impute
   imputegap.tools.utils


imputegap
=========

.. toctree::
   :maxdepth: 4

   imputegap
   _autosummary/imputegap.explainer
   modules  # Ensure modules is included here as well
