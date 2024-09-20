# Set these variables to point to your Armadillo and mlpack installation paths
ARMADILLO_INCLUDE = /mnt/c/armadillo/include
ARMADILLO_LIB = /mnt/c/armadillo/lib
MLPACK_INCLUDE = /mnt/c/mlpack/include
MLPACK_LIB = /mnt/c/mlpack/lib

# Compiler and flags
CXX = x86_64-w64-mingw32-g++
CXXFLAGS = -O3 -D ARMA_DONT_USE_WRAPPER -std=gnu++14 -fPIC -shared -fopenmp \
           -Wall -Wextra -pedantic -Wconversion -Wsign-conversion \
           -msse2 -msse3 -msse4 -msse4.1 -msse4.2
LDFLAGS = -lopenblas -larpack -larmadillo -lmlpack

# Target to build both the shared object and DLL
all: libAlgoCollection.so AlgoCollection.dll

# Rule to build the shared library for Linux
libAlgoCollection.so:
	g++ $(CXXFLAGS) -I$(ARMADILLO_INCLUDE) -L$(ARMADILLO_LIB) \
		-I$(MLPACK_INCLUDE) -L$(MLPACK_LIB) \
		-rdynamic -o libAlgoCollection.so \
		Algebra/Auxiliary.cpp Algorithms/TKCM.cpp Algorithms/ST_MVL.cpp \
		Algorithms/SPIRIT.cpp Algorithms/GROUSE.cpp Algorithms/NMFMissingValueRecovery.cpp \
		Algorithms/DynaMMo.cpp Algorithms/SVT.cpp Algorithms/ROSL.cpp \
		Algorithms/IterativeSVD.cpp Algorithms/SoftImpute.cpp Algorithms/CDMissingValueRecovery.cpp \
		Algebra/CentroidDecomposition.cpp Algebra/RSVD.cpp Stats/Correlation.cpp \
		shared/SharedLibFunctions.cpp $(LDFLAGS)

# Rule to build the DLL for Windows
AlgoCollection.dll:
	$(CXX) $(CXXFLAGS) -I$(ARMADILLO_INCLUDE) -L$(ARMADILLO_LIB) \
		-I$(MLPACK_INCLUDE) -L$(MLPACK_LIB) \
		-o AlgoCollection.dll \
		Algebra/Auxiliary.cpp Algorithms/TKCM.cpp Algorithms/ST_MVL.cpp \
		Algorithms/SPIRIT.cpp Algorithms/GROUSE.cpp Algorithms/NMFMissingValueRecovery.cpp \
		Algorithms/DynaMMo.cpp Algorithms/SVT.cpp Algorithms/ROSL.cpp \
		Algorithms/IterativeSVD.cpp Algorithms/SoftImpute.cpp Algorithms/CDMissingValueRecovery.cpp \
		Algebra/CentroidDecomposition.cpp Algebra/RSVD.cpp Stats/Correlation.cpp \
		shared/SharedLibFunctions.cpp $(LDFLAGS) \
		-Wl,--out-implib,libAlgoCollection.a -static-libgcc -static-libstdc++

# Rule to build without mlpack (for Linux)
nomlpack:
	g++ $(CXXFLAGS) -I$(ARMADILLO_INCLUDE) -L$(ARMADILLO_LIB) \
		-rdynamic -o libAlgoCollection.so \
		Algebra/Auxiliary.cpp Algorithms/TKCM.cpp Algorithms/ST_MVL.cpp \
		Algorithms/SPIRIT.cpp Algorithms/GROUSE.cpp Algorithms/DynaMMo.cpp \
		Algorithms/SVT.cpp Algorithms/ROSL.cpp Algorithms/IterativeSVD.cpp \
		Algorithms/SoftImpute.cpp Algorithms/CDMissingValueRecovery.cpp \
		Algebra/CentroidDecomposition.cpp Algebra/RSVD.cpp Stats/Correlation.cpp \
		shared/SharedLibFunctions.cpp -lopenblas -larpack -larmadillo

# Rule for a custom build ('brian') without mlpack, for a smaller subset of algorithms
brian:
	g++ $(CXXFLAGS) -I$(ARMADILLO_INCLUDE) -L$(ARMADILLO_LIB) \
		-rdynamic -o libAlgoCollection.so \
		Algebra/Auxiliary.cpp Algorithms/ST_MVL.cpp Algorithms/CDMissingValueRecovery.cpp \
		Algebra/CentroidDecomposition.cpp Algebra/RSVD.cpp Stats/Correlation.cpp \
		shared/SharedLibFunctions.cpp -lopenblas -larpack -larmadillo

# Rule to clean up build artifacts
clean:
	rm -f libAlgoCollection.so AlgoCollection.dll libAlgoCollection.a
