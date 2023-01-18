#!/bin/bash
#SBATCH -J final-optimization
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1

DIR=/home/pp22/pp22s52/final/cases
OUT=/dev/null
N=5000

# Compile code
make clean
make

# Generate testcase
echo -e "\n############### Generate Testcase: N=$N ###############"
./generate $DIR/t$N.in $N

# Run  testcase
for i in {CPU,GPU_Baseline,DP_Index,Parallel_MaxReduce,Two_Data_Per_Thread,Multiple_Data_Per_Thread,Unroll_Last_Warp,Unroll_All};
do
    echo -e "\n############### Optimization: $(python -c "print(' '.join('$i'.split('_')),end='')") ###############"
    echo -e "\n########## Time ##########"
    srun -N1 -n1 --gres=gpu:1 ./solution_$i $DIR/t$N.in $OUT
done

make clean