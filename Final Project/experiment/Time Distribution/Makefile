CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -lm -O3 -Wall
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
CXXFLAGS = $(CFLAGS)
TARGETS = generate solution

alls: $(TARGETS)

clean:
	rm -rf $(TARGETS)

generate: generate.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

solution: solution.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

