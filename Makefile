TARGET = Demo

SOURCES = Demo.cpp NUFFT3D.cpp 
HEADERS =
OBJECTS = $(SOURCES:.cpp=.o)

CXX = icpc
FLAGS = -std=c++11 -O3 -xHost -ipo -par-affinity=compact  -Wall -fopenmp -fp-model precise -DTIMING -qmkl #-par-affinity=compact -g
#LIB = -fopenmp -lfftw3f -lfftw3f_threads  -L/home/inspur/pac20/fftw3_threads/lib
#INC = -I/home/inspur/pac20/fftw3_threads/include -I/home/inspur/pac20/boost_1_74_0/include
LIB = -fopenmp -lfftw3xc_intel -lmkl_intel_ilp64  -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -L/home/inspur/zff_bs -L/home/inspur/intel/oneapi/mkl/2022.0.1/lib/intel64  -Wl,-rpath=/home/inspur/intel/oneapi/mkl/2022.0.1/lib/intel64 -Wl,-rpath=/home/inspur/intel/oneapi/compiler/2022.0.1/linux/compiler/lib/intel64_lin# -lfftw3f -lfftw3f_threads -L/home/inspur/pac20/fftw3_threads/lib
INC = -I/home/inspur/intel/oneapi/mkl/2022.0.1/include -I/home/inspur/intel/oneapi/mkl/2022.0.1/include/fftw/

################################################################################
################################################################################
################################################################################

%.o : %.cpp
	$(CXX) -c $(SOURCES) $(INC) $(FLAGS)

$(TARGET) : $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS) $(LIB)

.PHONY: clean
clean:	
	rm -rf $(OBJECTS) $(TARGET)

