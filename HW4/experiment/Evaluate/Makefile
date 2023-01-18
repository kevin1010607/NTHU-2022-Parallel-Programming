CC = mpicc
CXX = mpicxx
CFLAGS = -lpthread -O3
CXXFLAGS = $(CFLAGS) -std=c++17
TARGETS = mapreduce evaluate

alls: $(TARGETS)

clean:
	rm -f $(TARGETS)

mapreduce: mapreduce.cc
	$(CXX) $(CXXFLAGS) mapreduce.cc -o mapreduce

evaluate: evaluate.cc
	$(CXX) $(CXXFLAGS) evaluate.cc -o evaluate
