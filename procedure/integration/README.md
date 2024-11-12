![My Logo](https://www.naterscreations.com/imputegap/logo_imputegab.png)


# Installation for ImputeGAP

## System Requirements

To utilize **ImputeGAP**, the following prerequisites are necessary:

- Python version **3.12.0** or higher
- A **Unix-compatible environment** for execution

<br />


### Install WSL for Windows
To run your implementation in a Unix-compatible environment on Windows, we recommend installing **WSL (Windows Subsystem for Linux)**.

1. Check if **WSL** is already installed by typing `WSL` in the search menu.
2. If it is not installed, open **PowerShell** as Administrator (right-click the Start menu and select **Windows PowerShell (Admin)**).
3. Run the following command to install WSL:
   ```powershell
   wsl --install
   
3. This will install the latest version of WSL and a default Linux distribution (usually Ubuntu). After the installation is complete, you'll need to restart your computer.
<br><br>
*WSL can be selected in the IDE under the interpreter parameters.*

<br><br>

### Install Python 3.12.0

To use **ImputeGAP** effectively, ensure that your environment has **Python** version **3.12.0** or higher installed. Follow these steps to install or update Python in your Unix-compatible environment:

##### Step 1: Check Existing Python Version

Open your terminal and check the currently installed version of Python by running:

```bash
python3 --version
```
<br>

##### Step 2: Install Python
Update your package list and install the necessary dependencies for building Python:
```bash
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev python3-tk libopenblas0
```
<br>
Download Python 3.12.0 source code from the official Python website and extract it :

```bash
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
sudo tar xzf Python-3.12.0.tgz
```
<br>
Compile and install Python 12:

```bash
cd Python-3.12.0
sudo ./configure --enable-optimizations
sudo make altinstall
```
<br>
Verify the installation:

```bash
python3.12 --version
```


<br /><hr /><br />

## Installation



### Pip installation

To quickly install the latest version of **ImputeGAP** from the Python Package Index (PyPI), use the following command:

```bash
$ pip install imputegap
``` 

This will automatically install ImputeGAP along with its dependencies. Ensure that you are using Python 3.12.0 or higher in a Unix-compatible environment.

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