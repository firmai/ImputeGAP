###############
Getting Started
###############

.. _system-requirements:

System Requirements
~~~~~~~~~~~~~~~~~~~

ImputeGAP is compatible with Python>=3.10 (except 3.13) and Unix-compatible environment.

.. _installation:

Installation/Update
~~~~~~~~~~~~~~~~~~~

To install the latest version of ImputeGAP from PyPI, run the following command:

.. tabs::

    .. tab:: pip

        .. code-block:: bash

            pip install imputegap


    .. tab:: source

        If you would like to extend the library, you can install from source:

        .. code-block:: bash

            git init
            git clone https://github.com/eXascaleInfolab/ImputeGAP
            cd ./ImputeGAP
            pip install -e .


    .. tab:: env

        ImputeGAP is compatible with Python>=3.10 (except 3.13) and Unix-compatible environment.

        .. raw:: html

           <br>

        .. tabs::

            .. tab:: Windows:


                **Install WSL**

                To run ImputeGAP in a Unix-compatible environment on Windows, install the ``Windows Subsystem for Linux (WSL)``.

                1. Search for ``WSL`` in the Start menu to check if it's already installed.

                2. If not, open ``PowerShell`` as ``Administrator``.

                3. Run the following command:

                .. code-block:: bash

                    wsl --install

                4. Restart your computer once the installation completes.


                .. raw:: html

                   <br>


                **Prepare Python 3.12 Environment**

                To ensure a proper Python setup, we recommend creating a dedicated Python environment for the project. Python 3.12 is a suitable and supported choice.

                .. raw:: html

                   <br>

                *Step 1: Check Existing Python Version*

                Open your terminal and check the currently installed version of Python:

                .. code-block:: bash

                    python3 --version


                .. raw:: html

                   <br>

                *Step 2: Install Python 3.12*

                If needed, install Python 3.12 on your WSL system, follow these steps:

                1. Update your package list and install prerequisites:

                .. code-block:: bash

                    sudo apt-get update
                    sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
                    libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
                    tk-dev python3-tk libopenblas0 software-properties-common python3-pip


                3. Add the deadsnakes PPA and update:

                .. code-block:: bash

                    sudo add-apt-repository ppa:deadsnakes/ppa
                    sudo apt-get update


                4. Install Python 3.12:

                .. code-block:: bash

                    sudo apt-get install python3.12 python3.12-venv python3.12-dev


                5. Verify the installation:

                .. code-block:: bash

                    python3.12 --version


                .. raw:: html

                   <br>


                **Install Python 3.12 Environment**

                1. Create a virtual environment:

                .. code-block:: bash

                    python3.12 -m venv imputegap_env

                2. Activate the virtual environment:

                .. code-block:: bash

                    source imputegap_env/bin/activate


                3. Install ImputeGAP

                .. code-block:: bash

                    pip install imputegap


                .. raw:: html

                   <br><br>


            .. tab:: Linux:



                **Prepare Python 3.12 Environment**

                To ensure a proper Python setup, we recommend creating a dedicated Python environment for the project. Python 3.12 is a suitable and supported choice.

                .. raw:: html

                   <br>

                *Step 1: Check Existing Python Version**

                Open your terminal and check the currently installed version of Python:

                .. code-block:: bash

                    python3 --version


                .. raw:: html

                   <br>


                *Step 2: Install Python 3.12

                If needed, install Python 3.12 on your system, follow these steps:

                1. Update your package list and install prerequisites:

                .. code-block:: bash

                    sudo apt-get update
                    sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
                    libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
                    tk-dev python3-tk libopenblas0 software-properties-common python3-pip


                3. Add the deadsnakes PPA and update:

                .. code-block:: bash

                    sudo add-apt-repository ppa:deadsnakes/ppa
                    sudo apt-get update


                4. Install Python 3.12:

                .. code-block:: bash

                    sudo apt-get install python3.12 python3.12-venv python3.12-dev


                5. Verify the installation:

                .. code-block:: bash

                    python3.12 --version


                .. raw:: html

                   <br>


                **Install Python 3.12 Environment**

                1. Create a virtual environment:

                .. code-block:: bash

                    python3.12 -m venv imputegap_env


                2. Activate the virtual environment:

                .. code-block:: bash

                    source imputegap_env/bin/activate


                3. Install ImputeGAP

                .. code-block:: bash

                    pip install imputegap



                .. raw:: html

                   <br><br>




            .. tab:: MacOS:


                **Prepare Python 3.12 Environment**

                To ensure a proper Python setup, we recommend creating a dedicated Python environment for the project. Python 3.12 is a suitable and supported choice.

                1. Install Homebrew (if not already installed):

                .. code-block:: bash

                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


                2. Update Homebrew and install Python 3.12:

                .. code-block:: bash

                    brew update
                    brew install python@3.12


                3. Verify the installation:

                .. code-block:: bash

                    python3.12 --version


                .. raw:: html

                   <br>


                **Install Python 3.12 Environment**

                1. Create a virtual environment:

                .. code-block:: bash

                    python3.12 -m venv imputegap_env


                2. Activate the virtual environment:

                .. code-block:: bash

                    source imputegap_env/bin/activate


                3. Install ImputeGAP

                .. code-block:: bash

                    pip install imputegap




.. _docker:

Docker
~~~~~~

You can download the latest version of ImputeGAP with all dependencies pre-installed using Docker.


.. tabs::

    .. tab:: Windows

        Launch Docker and make sure it is running:

        .. code-block:: powershell

             docker version

        Pull the ImputeGAP Docker image:

        .. code-block:: powershell

             docker pull qnater/imputegap:0.0.8

        Run the Docker container:

        .. code-block:: powershell

            docker run -p 8888:8888 qnater/imputegap:0.0.8


    .. tab:: Ubuntu

        Launch Docker and make sure it is running:

        .. code-block:: powershell

             docker version

        Pull the ImputeGAP Docker image:

        .. code-block:: bash

            docker pull qnater/imputegap:0.0.8

        Run the Docker container:

        .. code-block:: bash

            docker run -p 8888:8888 qnater/imputegap:0.0.8

    .. tab:: MacOS

        Launch Docker and make sure it is running:

        .. code-block:: powershell

             docker version

        Pull the ImputeGAP Docker image:

        .. code-block:: bash

            docker pull --platform linux/x86_64 qnater/imputegap:0.0.8

        Run the Docker container:

        .. code-block:: bash

            docker run -p 8888:8888 qnater/imputegap:0.0.8


Troubleshooting
~~~~~~~~~~~~~~~

If you face any problems, please open an issue here: https://github.com/eXascaleInfolab/ImputeGAP/issues


