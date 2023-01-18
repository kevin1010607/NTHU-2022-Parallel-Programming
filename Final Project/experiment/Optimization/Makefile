CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -lm -O3 -Wall
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
CXXFLAGS = $(CFLAGS)
TARGETS = generate solution_CPU solution_GPU_Baseline solution_DP_Index solution_Parallel_MaxReduce solution_Two_Data_Per_Thread \
solution_Multiple_Data_Per_Thread solution_Unroll_Last_Warp solution_Unroll_All

alls: $(TARGETS)

clean:
	rm -rf $(TARGETS)

generate: generate.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

solution_CPU: solution_CPU.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

solution_GPU_Baseline: solution_GPU_Baseline.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

solution_DP_Index: solution_DP_Index.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

solution_Parallel_MaxReduce: solution_Parallel_MaxReduce.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

solution_Two_Data_Per_Thread: solution_Two_Data_Per_Thread.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

solution_Multiple_Data_Per_Thread: solution_Multiple_Data_Per_Thread.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

solution_Unroll_Last_Warp: solution_Unroll_Last_Warp.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

solution_Unroll_All: solution_Unroll_All.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?