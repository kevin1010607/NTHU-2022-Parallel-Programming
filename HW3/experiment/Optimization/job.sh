#!/bin/bash
#SBATCH -J hw3-2-optimization
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1

IN=/home/pp22/share/hw3-2/cases/p11k1
OUT=/dev/null

# Compile code
make clean
make

for i in {CPU,GPU_Baseline,Padding,Coalesced_Memory,Shared_Memory,Blocking_Factor_Tuning,Unroll};
do
    echo -e "\n############### Optimization: $(python -c "print(' '.join('$i'.split('_')),end='')") ###############"
    echo -e "\n########## Time ##########"
    srun -N1 -n1 --gres=gpu:1 ./hw3_$i $IN $OUT
done

make clean