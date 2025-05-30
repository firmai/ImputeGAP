=========
Notebooks
=========

ImputeGAP comes with ready-to-use Jupyter and Google Colab notebooks that illustrate key functionalities of the library. These notebooks provide hands-on examples for both the imputation pipeline and advanced analysis, helping users understand how to configure, run, and interpret the results of the system.

Jupyter Notebooks
-----------------

The following notebooks are available for local use or execution in Jupyter environments:

- `Jupyter Pipeline Creation <https://github.com/eXascaleInfolab/ImputeGAP/blob/refs/heads/main/imputegap/notebook/01_imputegap_imputation_pipeline.ipynb>`_
- `Jupyter Advanced Analytics <https://github.com/eXascaleInfolab/ImputeGAP/blob/refs/heads/main/imputegap/notebook/02_imputegap_advanced_analysis.ipynb>`_


.. raw:: html

   <br>

Google Colab Notebooks
----------------------

The following notebooks can be run directly in your browser using Google Colab. They are ideal for quick testing and exploration without requiring any local setup:

- `Google Colab Pipeline Creation <https://colab.research.google.com/drive/1Kq1_HVoCTWLtB1zyryR35opxXmaprztV?usp=sharing>`_
- `Google Colab Advanced Analytics <https://colab.research.google.com/drive/1iOzLtpZTA3KDoyIc-srw2eoX5soEmP8x?usp=sharing>`_


.. raw:: html

   <br>

Dockerized Notebooks
--------------------
To install ImputeGAP, pre-configured with Jupyter Notebook, as a Docker container:

.. tabs::

    .. tab:: Windows

        Launch Docker from desktop of terminal. To make sure it is running:

        .. code-block:: powershell

             docker version

        Pull the ImputeGAP Docker image:

        .. code-block:: powershell

             docker pull qnater/imputegap:0.0.8

        Run the Docker container:

        .. code-block:: powershell

            docker run -p 8888:8888 qnater/imputegap:0.0.8

        Open the following link:

        .. code-block:: powershell

            http://127.0.0.1:8888


    .. tab:: Linux

        Launch Docker from desktop of terminal. To make sure it is running:

        .. code-block:: powershell

             docker version

        Pull the ImputeGAP Docker image:

        .. code-block:: bash

            docker pull qnater/imputegap:0.0.8

        Run the Docker container:

        .. code-block:: bash

            docker run -p 8888:8888 qnater/imputegap:0.0.8

        Open the following link:

        .. code-block:: powershell

            http://127.0.0.1:8888

    .. tab:: MacOS

        Launch Docker from desktop of terminal. To make sure it is running:

        .. code-block:: powershell

             docker version

        Pull the ImputeGAP Docker image:

        .. code-block:: bash

            docker pull --platform linux/x86_64 qnater/imputegap:0.0.8

        Run the Docker container:

        .. code-block:: bash

            docker run -p 8888:8888 qnater/imputegap:0.0.8

        Open the following link:

        .. code-block:: powershell

            http://127.0.0.1:8888

