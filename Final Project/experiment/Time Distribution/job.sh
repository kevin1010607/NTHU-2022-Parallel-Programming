#!/bin/bash
#SBATCH -J final-time_distribution
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1

DIR=/home/pp22/pp22s52/final/cases
OUT=/dev/null
N=(100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)

# Compile code
make clean
make

# Generate testcase
for i in $(seq 0 11);
do
    echo -e "\n############### Generate Testcase: N=${N[$i]} ###############"
    ./generate $DIR/t${N[$i]}.in ${N[$i]}
done

# Run  testcase
for i in $(seq 0 11);
do
    echo -e "\n############### Run Testcase: N=${N[$i]} ###############"
    echo -e "\n########## Time ##########"
    srun -N1 -n1 --gres=gpu:1 ./solution $DIR/t${N[$i]}.in $OUT
    echo -e "\n########## Metric ##########"
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof ./solution $DIR/t${N[$i]}.in $OUT > $OUT
done

make clean