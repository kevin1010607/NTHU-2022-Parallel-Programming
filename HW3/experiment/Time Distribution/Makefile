CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -lm -O3
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
CXXFLAGS = $(CFLAGS)
TARGETS = hw3-2

alls: $(TARGETS)

clean:
	rm -f $(TARGETS)

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?