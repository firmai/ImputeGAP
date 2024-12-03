<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# ImputeGAP Integration Module

## Getting Started

### Initializing a Git Repository

To begin using the ImputeGAP library, initialize a Git repository and clone the project from GitHub:

```bash
$ git init
$ git clone https://github.com/eXascaleInfolab/ImputeGAP
$ cd ./ImputeGAP
``` 

<br>

### Python Integration Steps

1) Navigate to the ```./imputegap/algorithms``` directory. 


2) Create a new file by copying ```mean_inpute.py``` and rename it to reflect your algorithm.


3) Replace the logic section under ```# logic``` with your algorithm’s implementation.


4) Adapt the parameters as needed, ensuring to follow the ```TimeSeries``` object structure for input and return a ```numpy.ndarray``` matrix. Refer to the docstring of the template for detailed guidance.


5) Update  ```./imputegap/recovery/imputation.py```:
   1) Add a function to call your new algorithm by copying the ```class MeanImpute(BaseImputer)```and modifying it to suit your requirements.
   2) You can add it into the corresponding category of algorithms.


6) Perform imputation as needed.

<br />

### C++ Integration
1) Navigate to the ```./imputegap/algorithms``` directory.


2) If not already done, convert your CPP/H files into a shared object format  (```.so```) and place them in the  ```imputegap/algorithms/lib``` folder.
   1) Go to ```./imputegap/wrapper/AlgoCollection```  and update the Makefile. Copy commands from ```libSTMVL.so``` or modify them as needed.
   2) Optionally, copy your C++ project files into the directory.
   3) Generate the ```.so``` file using the ```make``` command:
      ```
      make your_lib_name
      ```
   4) Optional: To include the .so file in the "in-built" directory, open a command line, navigate to the root directory, and execute the library build process:
      ```
      rm -rf dist/
      python setup.py sdist bdist_wheel
      ```


3) Rename ```cpp_integration.py```  to reflect your algorithm’s name.


4) Modify the ```native_algo()``` function: 
   1) Update the shared object parameter to match your shared library.
   2) Convert input parameters to the appropriate C++ types and pass them to your shared object methods.
   3) Convert the imputed matrix back to a numpy format.


5) Adapt the template method ```your_algo.py``` with the appropriate parameters, ensuring compatibility with the ```TimeSeries``` object and a ```numpy.ndarray``` return type.


6) Adapt the  ```./imputegap/recovery/imputation.py```:
   1) Add a function to call your new algorithm by copying and modifying ```class MeanImpute(BaseImputer)``` as needed. You can copy-paste the class into the corresponding category of algorithms.


7) Perform imputation as needed.
 
<br /><br />


#### Example: CDRec Integration

######  Shared Object Generation

1) Modify the Makefile:
```
libCDREC.so: 
    g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_cdrec.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Stats/Correlation.cpp Algorithms/CDMissingValueRecovery.cpp  Algebra/Auxiliary.cpp \
	Algebra/CentroidDecomposition.cpp  shared/SharedLibCDREC.cpp \
	-lopenblas -larpack -lmlpack
```


2) Generate the shared library:

```
make libCDREC.so
```


3) Place the generated ```.so``` file in ```imputegap/algorithms/lib```



4) Optional: To include the .so file in the "in-built" directory, open a command line, navigate to the root directory, and execute the library build process:
```
rm -rf dist/
python setup.py sdist bdist_wheel
```
   
<br> 

######  Wrapper 

1) In ```imputegap/algorithms/cpp_integration.py```, update the function name and parameter count, and ensure the ```.so``` file matches:
```
def native_cdrec(__py_matrix, __py_rank, __py_epsilon, __py_iterations):

    shared_lib = utils.load_share_lib("lib_cdrec.so") # in-build files
    # shared_lib = utils.load_share_lib("./your_path/lib_cdrec.so") # external files
```
<br> 

2) Convert variables to corresponding C++ types:
```
    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_rank >= 0);
    assert (__py_rank < __py_m);
    assert (__py_epsilon > 0);
    assert (__py_iterations > 0);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __ctype_rank = __native_c_types_import.c_ulonglong(__py_rank);
    __ctype_epsilon = __native_c_types_import.c_double(__py_epsilon);
    __ctype_iterations = __native_c_types_import.c_ulonglong(__py_iterations);
    
    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = __marshal_as_native_column(__py_matrix);
```
<br> 

3) Call the C++ algorithm with the required parameters:
```
    shared_lib.cdrec_imputation_parametrized(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_rank, __ctype_epsilon, __ctype_iterations);
```
<br> 

4) Convert the imputed matrix back to ```numpy```:
```
    __py_imputed_matrix = __marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;
```
<br>


######  Method Implementation

1) In ```imputegap/algorithms/cpp_integration.py```, create or adapt a generic method for your needs:

```
def cdrec(contamination, truncation_rank, iterations, epsilon, logs=True, lib_path=None):
   
    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    imputed_matrix = native_cdrec(contamination, truncation_rank, epsilon, iterations)

    end_time = time.time()

    if logs:
        print(f"\n\t\t> logs, imputation cdrec - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return imputed_matrix
```
<br> 

######  Imputer Class

1) Add your algorithm to the catalog in```./imputegap/recovery/imputation.py```


2) Copy and modify ```class MeanImpute(BaseImputer)```  to fit your requirements:
   1) You can copy-paste the class into the corresponding category of algorithm.

```
class MatrixCompletion:
    class CDRec(BaseImputer):
        algorithm = "cdrec"

        def impute(self, user_defined=True, params=None):
            
            self.imputed_matrix = cdrec(contamination=self.infected_matrix, truncation_rank=rank, iterations=iterations, epsilon=epsilon, logs=self.logs)
            
            return self
```

