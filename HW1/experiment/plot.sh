#!/bin/bash
#SBATCH -J hw1-plot
#SBATCH -N 4
#SBATCH -n 48

N=536869888
IN=/home/pp22/share/hw1/testcases/40.in
OUT=/dev/null

# Compile code
make clean
make hw1_time

mkdir -p ./image
cat /dev/null > data.txt

# Single node (1~12 process)
for ((i=1;i<=12;i++)) do
    echo "########## Single node - $i process ##########"
    srun -N1 -n$i ./hw1_time $N $IN $OUT >> data.txt
done

# Multi node (1~4 node, 12 process per node)
for ((i=1;i<=4;i++)) do
    echo "########## Multi node - $i node ##########"
    srun -N$i -n$(expr $i \* 12) ./hw1_time $N $IN $OUT >> data.txt
done

# Plot
echo "########## Plot ##########"
python plot.py < data.txt
# rm data.txt

make clean