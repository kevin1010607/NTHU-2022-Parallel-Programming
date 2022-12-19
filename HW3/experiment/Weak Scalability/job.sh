#!/bin/bash
#SBATCH -J hw3-2-weak_scalability
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -G 2

IN1=/home/pp22/share/hw3-2/cases/p25k1
IN2=/home/pp22/share/hw3-3/cases/p35k1
OUT=/dev/null

# Compile code
make clean
make

echo -e "\n############### Weak Scalability: 1 GPU ###############"
echo -e "\n########## Time ##########"
srun -N1 -n1 --gres=gpu:1 ./hw3-2 $IN1 $OUT

echo -e "\n############### Weak Scalability: 2 GPU ###############"
echo -e "\n########## Time ##########"
srun -N1 -n2 --gres=gpu:2 ./hw3-3 $IN2 $OUT

make clean