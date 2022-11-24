#!/bin/bash
#SBATCH -J hw1-plot
#SBATCH -N 4
#SBATCH -n 48

# Compile code
n=536869888
make clean
make

mkdir -p ./image
cat /dev/null > data.txt

# Single node (1~12 process)
for ((i=1;i<=12;i++)) do
    srun -N 1 -n $i ./test $n test.in test.out >> data.txt
done

# Multi node (1~4 node, 12 process per node)
for ((i=1;i<=4;i++)) do
    srun -N $i -n $(expr $i \* 12) ./test $n test.in test.out >> data.txt
done

# Plot
python plot.py < data.txt
# rm data.txt