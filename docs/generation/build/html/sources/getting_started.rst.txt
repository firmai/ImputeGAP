===============
Getting Started
===============

.. _system-requirements:

System Requirements
-------------------

The following prerequisites are required to use ImputeGAP:

- Python version 3.10 / 3.11 / 3.12
- Unix-compatible environment for execution

To create and set up an environment with Python 3.12, please refer to the `installation guide <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/procedure/installation>`_.

.. _installation:

Installation
-------------------

**Pip Installation**

To quickly install the latest version of ImputeGAP along with its dependencies from the Python Package Index (PyPI), run the following command:

.. code-block:: bash

    pip install imputegap

**Local Installation**

To modify the code of ImputeGAP or contribute to its development, you can install the library from source:

1) Initialize a Git repository and clone the project from GitHub:

.. code-block:: bash

    git init
    git clone https://github.com/eXascaleInfolab/ImputeGAP
    cd ./ImputeGAP

2) Once inside the project directory, run the following command to install the package in editable mode:

.. code-block:: bash

    pip install -e .
