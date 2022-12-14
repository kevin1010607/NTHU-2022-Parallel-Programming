#!/bin/bash
#SBATCH -J hw2-plot
#SBATCH -N 4
#SBATCH -n 48

IN=/home/pp22/share/hw2/testcases/strict23.txt
OUT=/dev/null

# Compile code
make clean
make hw2a_time
make hw2b_time

mkdir -p ./image
cat /dev/null > data.txt

# Single node for pthread (1~12 thread)
for ((i=1;i<=12;i++)) do
    echo "########## Pthread - $i thread ##########"
    srun -n1 -c$i ./hw2a_time $OUT $(cat $IN) $i >> data.txt
done

# Multi node for hybrid (1~4 node, 1 process per node, 12 thread per process)
for ((i=1;i<=4;i++)) do
    echo "########## Hybrid - $i node ##########"
    srun -N$i -n$(expr $i \* 1) -c12 ./hw2b_time $OUT $(cat $IN) 12 >> data.txt
done

# Plot
echo "########## Plot ##########"
python plot.py < data.txt
# rm data.txt

make clean