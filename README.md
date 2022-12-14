# NTHU 2022 Parallel Programming

## HW1 - Odd-Even Sort

### Goal
- This assignment helps you get familiar with MPI by implementing a parallel odd-even sort algorithm.
- Experimental evaluations on your implementation are required to guide you analyze the performance and scalability of a parallel program.
- You are also encouraged to explore any performance optimization and parallel computing skills in order to receive a higher score.
### Implementation
- You are required to implement a parallel version of odd-even sort under the given restrictions.
- Your goal is to optimize the performance and reduce the total execution time.

## HW2 - Mandelbrot Set

### Goal
- You are asked to parallelize the sequential Mandelbrot Set program.
- Get familiar with thread programming using Pthread and OpenMP.
- Combine process and thread to implement a hybrid parallelism solution.
- Understand the importance of load balance.
### Implementation
- You are asked to parallelize the sequential Mandelbrot Set program by implementing the following two versions:
  - `pthread`: Single node shared memory programming using Pthread.
    - This program only needs to be run on a single node.
  - `hybrid`: Multi-node hybrid parallelism programming using MPI + OpenMP.
    - This program must be run across multiple nodes.
    - MPI processes are used to balance tasks among nodes, and OpenMP threads are used to perform computations.
    - Pthread library could also be used to create additional threads for communications.

## HW3 - All-Pairs Shortest Path

### Goal
- This assignment helps you manage to solve the all-pairs shortest path problem with CPU threads and then further accelerate the program with CUDA accompanied by Blocked Floyd-Warshall algorithm.
- We encourage you to optimize your program by exploring different optimizing strategies for performance points.
### Implementation
- You are asked to implement 3 versions of programs that solve the all-pairs shortest path problem.
  - CPU version (hw3-1)
    - You are required to use threading to parallelize the computation in your program.
    - You can choose any threading library or framework you like (pthread, std::thread, OpenMP, Intel TBB, etc).
    - You can choose any algorithm to solve the problem.
    - You must implement the shortest path algorithm yourself.
  - Single-GPU version (hw3-2)
    - Should be optimized to get the performance points.
  - Multi-GPU version (hw3-3)
    - Must use 2 GPUs.

## HW4 - MapReduce

### Goal
- This assignment provides an opportunity for you to practice your parallel programming skills by implementing the scheduling and parallel programming model of the well-known big data processing framework, MapReduce.
### Implementation
- You are asked to implement a parallel program that mimics the data locality-aware scheduling policy and the functional level programming model of MapReduce.
- You will implement the parallel program using MPI and Pthread library. The jobtracker(scheduler) and tasktrackers(workers) are implemented as MPI processes, and threads are used for executing computing tasks and IO.
- The jobtracker is responsible for generating the map tasks, reducing tasks of a MapReduce job and following the data-locality scheduling principle to dispatch tasks on worker nodes for execution.
- Each node runs a tasktracker which is responsible for creating and managing a set of mapper and reducer threads to execute the receiving map tasks and reduce tasks and outputs the intermediate and final output files.
- We do NOT consider worker nodes to join, leave or fail during the job execution.
- You are required to implement MapReduce system architecture, programming model, and scheduling algorithm described in Section 3, 4 and 5, respectively.
- You are required to implement a WordCount sample code to demonstrate your implementation.
- All the codes should be compiled into a single MPI program, and you should make sure the program terminates properly after all the computing tasks are completed.
- Performance is not the primary concern in this assignment, but you are still encouraged to improve the code efficiently.

## Lab1 - Pixels in circle (MPI)

### Problem
- Suppose we want to draw a filled circle of radius r on a 2D monitor, how many pixels will be filled?
- We fill a pixel when any part of the circle overlaps with the pixel.
- We also assume that the circle center is at the boundary of 4 pixels.
### Implementation
- Parallelize the calculation using MPI.

## Lab2 - Pixels in circle (Pthread & OpenMP)

### Problem
- Same as Lab1.
### Implmentation
- We are going to approximate pixels using pthread, OpenMP and hybrid of MPI and OpenMP in this lab.

## Lab3 - Edge detection (Cuda)

### Problem
- Identifying points in a digital image at which the image brightness changes sharply.
- Sobel Operator: Used in image processing and computer vision, particularly within edge detection algorithms.
- Uses two 5 x 5 kernels gx, gy which are convolved with the original image to calculate approximations of the derivatives one for horizontal changes, and one for vertical.
### Implmentation
- Parallelize the calculation using GPU and cuda.

## Lab4 - Edge detection (Cuda Advance)

### Problem
- Same as Lab3
### Implementation
- Further optimize the code with following techniques
  - Coalesced Memory Access
  - Lower Precision
  - Shared Memory

## Lab5

## Lab6

## Final Project
