CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -lm -O3
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
hw3_CPU: CFLAGS += -fopenmp -msse2 -msse3 -msse4
CXXFLAGS = $(CFLAGS)
TARGETS = hw3_CPU hw3_GPU_Baseline hw3_Padding hw3_Coalesced_Memory hw3_Shared_Memory hw3_Blocking_Factor_Tuning hw3_Unroll

alls: $(TARGETS)

clean:
	rm -f $(TARGETS)

hw3_CPU: hw3_CPU.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3_GPU_Baseline: hw3_GPU_Baseline.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3_Padding: hw3_Padding.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3_Coalesced_Memory: hw3_Coalesced_Memory.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3_Shared_Memory: hw3_Shared_Memory.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3_Blocking_Factor_Tuning: hw3_Blocking_Factor_Tuning.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3_Unroll: hw3_Unroll.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?