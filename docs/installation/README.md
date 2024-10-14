![My Logo](https://www.naterscreations.com/imputegap/logo_imputegab.png)


# Installation for ImputeGAP

## Requirements
In order to use **ImputeGAP**, you must have Python **3.12.0** or higher and run your code in a **Unix-compatible environment**.
<br><br>


### Install WSL for Windows
To run your implementation in a Unix-compatible environment on Windows, we recommend you install **WSL (Windows Subsystem for Linux)**.

0. Check if you already have installed **WSL**, by typing `WSL` in the search menu.
1. If it is not installed, open **PowerShell** as Administrator (right-click the Start menu and select **Windows PowerShell (Admin)**).
2. Run the following command to install WSL:
   ```powershell
   wsl --install
   
3. This will install the latest version of WSL and a default Linux distribution (usually Ubuntu). After the installation, you'll need to restart your computer.
<br><br>
*WSL can be selected in the IDE in the interpreter parameters.*
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
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev
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
To install in local ImputeGAP, download the package from GitHub and run the command : 

```pip install -e .``` 



<br /><hr /><br />