CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -lm -O3
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
CXXFLAGS = $(CFLAGS)
TARGETS = hw3-2-16 hw3-2-32 hw3-2-64

alls: $(TARGETS)

clean:
	rm -f $(TARGETS)

hw3-2-16: hw3-2-16.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?	

hw3-2-32: hw3-2-32.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-64: hw3-2-64.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?