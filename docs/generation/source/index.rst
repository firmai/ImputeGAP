ImputeGAP Documentation
=======================

ImputeGAP is a comprehensive Python library for imputation of missing values in  time series data. It implements user-friendly APIs to easily visualize, analyze, and repair your own time series datasets. The library supports a diverse range of imputation methods and modular missing data simulation catering to datasets with varying characteristics. ImputeGAP includes extensive customization options, such as automated hyperparameter tuning, benchmarking, explainability, downstream evaluation, and compatibility with popular time series frameworks.


In detail, the library provides:
    - Access to commonly used datasets in time series research (`Datasets <datasets.html>`_).
    - Automated preprocessing with built-in methods for normalizing time series (`Preprocessing <preprocessing.html>`_).
    - Configurable contamination module that simulates real-world missingness patterns (`Patterns <patterns.html>`_).
    - Parameterizable state-of-the-art time series imputation algorithms (`Algorithms <algorithms.html>`_).
    - Benchmarking to foster reproducibility in time series imputation (`Benchmark <benchmark.html>`_).
    - Modular tools to analyze the behavior of imputation algorithms and assess their impact on key downstream tasks in time series analysis (`Downstream Evaluation <downstream.html>`_).
    - Fine-grained analysis of the impact of time series features on imputation results (`Explainer <explainer.html>`_).
    - Plug-and-play integration of new datasets and algorithms in various languages such as Python, C++, Matlab, Java, and R.


.. raw:: html

   <br><br>

.. _data-format:

Data Format
-----------

Please ensure that your data satisfies the following criteria:

.. note::

    - 2D Matrix: columns are the series and rows are the timestamps
    - Column separator: empty space
    - Row separator: newline
    - Missing values are NaN
    - Data output uses ``numpy.ndarray``



.. raw:: html

   <br><br>


.. _get_started:

Get Started
___________

.. raw:: html

   <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">

       <div style="background-color: #181818; padding: 15px; border-radius: 8px;">
       <h3><a href="getting_started.html" style="text-decoration: none; color: #2e86c1;">🚀 Installation</a></h3>
       <p style="color: #CCCCCC;">Read the guide on how to install <strong>ImputeGAP</strong> on your system.</p>
       </div>

       <div style="background-color: #181818; padding: 15px; border-radius: 8px;">
       <h3><a href="tutorials.html" style="text-decoration: none; color: #2e86c1;">📖 Tutorials</a></h3>
       <p style="color: #CCCCCC;">Check the tutorials to learn how to use <strong>ImputeGAP</strong> efficiently.</p>
       </div>

       <div style="background-color: #181818; padding: 15px; border-radius: 8px;">
       <h3><a href="imputegap.html" style="text-decoration: none; color: #2e86c1;">📦 API</a></h3>
       <p style="color: #CCCCCC;">Find the main API for each submodule in the index.</p>
       </div>

       <div style="background-color: #181818; padding: 15px; border-radius: 8px;">
       <h3><a href="algorithms.html" style="text-decoration: none; color: #2e86c1;">🧠 Algorithms</a></h3>
       <p style="color: #CCCCCC;">Explore the core algorithms used in <strong>ImputeGAP</strong>.</p>
       </div>

   </div><br><br>


.. _citing:

Citing
------

If you use ImputeGAP in your research, please cite the paper:

.. code-block:: bash

    @article{nater2025imputegap,
      title = {ImputeGAP: A Comprehensive Library for Time Series Imputation},
      author = {Nater, Quentin and Khayati, Mourad and Pasquier, Jacques},
      year = {2025},
      eprint = {2503.15250},
      archiveprefix = {arXiv},
      primaryclass = {cs.LG},
      url = {https://arxiv.org/abs/2503.15250}
    }


.. raw:: html

   <br><br>


.. _contributors:

Contributors
____________

.. list-table::
   :widths: 100 100
   :align: center
   :header-rows: 0

   * - .. image:: _img/quentin_nater.png
          :alt: Quentin Nater - ImputeGAP
          :width: 100px
          :height: 100px
          :align: center
          :target: https://exascale.info/members/quentin-nater/
     - .. image:: _img/mourad_khayati.png
          :alt: Mourad Khayati - ImputeGAP
          :width: 100px
          :height: 100px
          :align: center
          :target: https://exascale.info/members/mourad-khayati/

   * - Quentin Nater
     - Mourad Khayati








.. toctree::
   :maxdepth: 0
   :caption: Contents:
   :hidden:


   index
   getting_started
   tutorials
   datasets
   preprocessing
   patterns
   algorithms
   benchmark
   downstream
   explainer
   GitHub Repository <https://github.com/eXascaleInfolab/ImputeGAP/>
   PyPI Repository <https://pypi.org/project/imputegap/>
   imputegap




