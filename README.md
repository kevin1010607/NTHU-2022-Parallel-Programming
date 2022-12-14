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

## HW3

## HW4

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

## Lab4

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
