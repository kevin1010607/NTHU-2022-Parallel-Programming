CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a_time: CFLAGS += -pthread
hw2b_time: CC = mpicc
hw2b_time: CXX = mpicxx
hw2b_time: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2a_time hw2b_time

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
