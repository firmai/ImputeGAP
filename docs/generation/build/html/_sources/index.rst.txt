ImputeGAP documentation
=======================

ImputeGAP is a unified framework for imputation algorithms that provides a narrow-waist interface between algorithm evaluation and parameterization for datasets issued from various domains ranging from neuroscience, medicine, climate to energy.

The interface provides advanced imputation algorithms, construction of various missing values patterns, and different evaluation metrics. In addition, the framework offers support for AutoML parameterization techniques, feature extraction, and, potentially, analysis of feature impact using SHAP. The framework should allow a straightforward integration of new algorithms, datasets, and metrics.

.. _data-format:

Data Format
-----------

If you use your own datasets, please make sure your data satisfies the following conditions.

.. note::

    - **Data Type:** Must be a ``numpy.ndarray``.
    - **Structure:** The data should be a **2D matrix**.
    - **Rows:** Each row represents a set of values (e.g., **Timestamp #1**).
    - **Columns:** Each column corresponds to a different series (e.g., **Temperature**).
    - **Row Separator:** Rows should be separated by a **carriage return** (``"\n"``).
    - **Column Separator:** Values within columns should be separated by a **space** (``" "``).
    - **Missing Values:** Can be detected using ``numpy.isnan()``.
    - **Shape:** The numpy shape of the matrix imported is (series, values)
    - **Example:** `Dataset Example <https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/dataset/test.txt>`_ - The shape of the time series is (10, 25), the number of series is 10 and the number of values is 25.


.. _algorithms:

Algorithms
----------

.. list-table::
   :header-rows: 1

   * - **ALGORITHMS**
     - **FAMILIES**
     - **CONF**
   * - CDRec
     - Matrix Completion
     - KAIS'20
   * - IterativeSVD
     - Matrix Completion
     - BIOINFORMATICS'01
   * - GROUSE
     - Matrix Completion
     - PMLR'16
   * - ROSL
     - Matrix Completion
     - CVPR'14
   * - SPIRIT
     - Matrix Completion
     - VLDB'05
   * - SoftImpute
     - Matrix Completion
     - JMLR'10
   * - SVT
     - Matrix Completion
     - SIAM J. OPTIM'10
   * - TRMF
     - Matrix Completion
     - NeurIPS'16
   * - ST-MVL
     - Pattern Search
     - IJCAI'16
   * - DynaMMo
     - Pattern Search
     - KDD'09
   * - TKCM
     - Pattern Search
     - EDBT'17
   * - IIM
     - Machine Learning
     - ICDE '19
   * - XGBI
     - Machine Learning
     - KDD'16
   * - Mice
     - Machine Learning
     - Statistical Software'11
   * - MissForest
     - Machine Learning
     - BioInformatics'11
   * - KNNImpute
     - Statistics
     - native
   * - Interpolation
     - Statistics
     - native
   * - Min Impute
     - Statistics
     - native
   * - Mean Impute
     - Statistics
     - native
   * - Mean Impute By Series
     - Statistics
     - native
   * - MRNN
     - Deep Learning
     - IEEE Trans on BE'19
   * - BRITS
     - Deep Learning
     - NeurIPS'18
   * - DeepMVI
     - Deep Learning
     - PVLDB'21
   * - MPIN
     - Deep Learning
     - PVLDB'24
   * - PriSTI
     - Deep Learning
     - ICDE'23
   * - MissNet
     - Deep Learning
     - KDD'24
   * - GAIN
     - Deep Learning
     - ICML'18
   * - GRIN
     - Deep Learning
     - ICLR'22
   * - BayOTIDE
     - Deep Learning
     - PMLR'24
   * - HKMF-T
     - Deep Learning
     - TKDE'21



.. _api:

API
---
.. autosummary::
   imputegap.recovery.manager
   imputegap.recovery.imputation
   imputegap.recovery.optimization
   imputegap.recovery.explainer
   imputegap.recovery.evaluation
   imputegap.recovery.benchmark
   imputegap.tools.utils


.. _tree:

TREE
----

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   getting_started
   datasets
   tutorials
   references
   GitHub Repository <https://github.com/eXascaleInfolab/ImputeGAP/>
   PyPI Repository <https://pypi.org/project/imputegap/>
   imputegap




