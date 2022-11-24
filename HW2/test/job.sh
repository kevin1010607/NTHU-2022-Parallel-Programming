#!/bin/bash
#SBATCH -J hw2-plot
#SBATCH -N 4
#SBATCH -n 48

# Compile code
make clean
make

mkdir -p ./image
cat /dev/null > data.txt

# Single node for pthread (1~12 thread)
for ((i=1;i<=12;i++)) do
    srun -n 1 -c $i ./test_a output.png $(cat test.txt) $i >> data.txt
done

# Multi node for hybrid (1~4 node, 1 process per node, 12 thread per process)
for ((i=1;i<=4;i++)) do
    srun -N $i -n $(expr $i \* 1) -c 12 ./test_b output.png $(cat test.txt) 12 >> data.txt
done

# Plot
python plot.py < data.txt
# rm data.txt