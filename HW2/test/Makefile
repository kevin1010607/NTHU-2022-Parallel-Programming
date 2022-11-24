CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3
test_a: CFLAGS += -pthread
test_b: CC = mpicc
test_b: CXX = mpicxx
test_b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = test_a test_b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
