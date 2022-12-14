CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -lm -O3
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
hw3-1: CFLAGS += -fopenmp -msse2 -msse3 -msse4
hw3-3: NVFLAGS += -Xcompiler="-fopenmp"
CXXFLAGS = $(CFLAGS)
TARGETS = hw3-1 hw3-2 hw3-3

alls: $(TARGETS)

clean:
	rm -f $(TARGETS)

seq: seq.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?