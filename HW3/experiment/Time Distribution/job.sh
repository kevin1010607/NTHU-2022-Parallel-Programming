#!/bin/bash
#SBATCH -J hw3-2-time_distribution
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1

DIR=/home/pp22/share/hw3-2/cases/
IN=(c21.1 p11k1 p15k1 p20k1 p25k1)
N=(5000 11000 15000 20000 25000)
OUT=/dev/null

# Compile code
make clean
make

for i in $(seq 0 4);
do
    echo -e "\n############### Time Distribution: ${IN[$i]} - N=${N[$i]} ###############"
    echo -e "\n########## Time ##########"
    srun -N1 -n1 --gres=gpu:1 ./hw3-2 $DIR${IN[$i]} $OUT
    echo -e "\n########## Metric ##########"
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof ./hw3-2 $DIR${IN[$i]} $OUT > $OUT
done

make clean