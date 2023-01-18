CC = mpicc
CXX = mpicxx
CFLAGS = -lpthread -O3
CXXFLAGS = $(CFLAGS) -std=c++17
TARGETS = mapreduce

alls: $(TARGETS)

clean:
	rm -f $(TARGETS)

mapreduce: mapreduce.cc
	$(CXX) $(CXXFLAGS) mapreduce.cc -o mapreduce
