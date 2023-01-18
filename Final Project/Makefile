CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -lm -O3 -Wall
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
CXXFLAGS = $(CFLAGS)
TARGETS = format generate solution_CPU solution_GPU

alls: $(TARGETS)

clean:
	rm -rf $(TARGETS)

format: format.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

generate: generate.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

solution_CPU: solution_CPU.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

solution_GPU: solution_GPU.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

