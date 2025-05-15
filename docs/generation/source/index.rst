ImputeGAP Documentation
=======================

ImputeGAP is a comprehensive Python library for imputation of missing values in  time series data. It implements user-friendly APIs to easily visualize, analyze, and repair time series datasets. The library supports a diverse range of imputation algorithms and modular missing data simulation catering to datasets with varying characteristics. ImputeGAP includes extensive customization options, such as automated hyperparameter tuning, benchmarking, explainability, and downstream evaluation.


In detail, the library provides:
    - Access to commonly used datasets in the time series imputation field (`Datasets <datasets.html>`_).
    - Configurable contamination module that simulates real-world missingness patterns (`Patterns <patterns.html>`_).
    - Parameterizable state-of-the-art time series imputation algorithms (`Algorithms <algorithms.html>`_).
    - Extensive benchmarking to compare the performance of imputation algorithms (`Benchmark <tutorials.html#benchmark>`_).
    - Modular tools to assess the impact of imputation on key downstream tasks (`Downstream <tutorials.html#downstream>`_).
    - Fine-grained analysis of the impact of time series features on imputation results (`Explainer <tutorials.html#explainer>`_).
    - Seamless integration of new algorithms in Python, C++, Matlab, Java, and R (`Contributing <contributing.html>`_).

.. raw:: html

   <br>

.. note::

    If you like our library, please add a ⭐ in our `GitHub repository <https://github.com/eXascaleInfolab/ImputeGAP/>`_.



.. raw:: html

   <br>

.. _data-format:

Data Format
-----------
Please ensure that your input data satisfies the following format:

.. note::

    - Columns are the series' values, and rows are the timestamps
    - Column separator: empty space
    - Row separator: newline
    - Missing values are NaNs



.. raw:: html

   <br>


.. _get_started:

Get Started
-----------

.. raw:: html

   <script>
      function applyTheme(e)
      {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      }

      const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
      applyTheme(darkQuery); // Apply on load
      darkQuery.addEventListener('change', applyTheme); // React to changes
   </script>

   <style>
        [data-theme="dark"] .card
        {
            padding: 15px;
            border-radius: 8px;
            background-color: #181818;
        }

        [data-theme="dark"] .card p
        {
            color: #CCCCCC;
        }

        [data-theme="dark"] .card h3 a
        {
            color: #2e86c1;
            text-decoration: none;
        }

        [data-theme="light"] .card
        {
            padding: 15px;
            border-radius: 8px;
            background-color: #f8f9fa;
        }

        [data-theme="light"] .card p
        {
            color: #333333;
        }

        [data-theme="light"] .card h3 a
        {
            color: #2e86c1;
            text-decoration: none;
        }
   </style>


   <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">

      <div class="card">
        <h3><a href="getting_started.html">🚀 Installation</a></h3>
        <p>Read the guide on how to install ImputeGAP on your system.</p>
      </div>

      <div class="card">
        <h3><a href="tutorials.html">📖 Tutorials</a></h3>
        <p>Check the tutorials to learn how to use ImputeGAP efficiently.</p>
      </div>

      <div class="card">
        <h3><a href="imputegap.html">📦 API</a></h3>
        <p>Find the main API for each submodule in the index.</p>
      </div>

      <div class="card">
        <h3><a href="algorithms.html">🧠 Algorithms</a></h3>
        <p>Explore the core algorithms used in ImputeGAP.</p>
      </div>

    </div><br>




.. _citing:

Citing
------

If you use ImputeGAP in your research, please cite these papers:

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

   <br>

.. code-block:: bash

    @article{nater2025kdd,
      title = {A Hands-on Tutorial on Time Series Imputation with ImputeGAP},
      author = {Nater, Quentin and Khayati, Mourad and Cudré-Mauroux, Philippe},
      year = {2025},
      booktitle = {SIGKDD Conference on Knowledge Discovery and Data Mining (To Appear)},
      series = {KDD2025}
    }

.. raw:: html

   <br>


.. toctree::
   :maxdepth: 0
   :caption: Contents:
   :hidden:


   index
   getting_started
   tutorials
   notebook
   datasets
   patterns
   algorithms
   contributing
   about_us
   imputegap


