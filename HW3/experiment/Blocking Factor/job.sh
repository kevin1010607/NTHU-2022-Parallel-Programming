#!/bin/bash
#SBATCH -J hw3-2-blocking_factor
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1

IN=/home/pp22/share/hw3-2/cases/c21.1
OUT=/dev/null
NVPROF_METRIC=inst_integer,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput

# Compile code
make clean
make

for i in {16,32,64};
do
    echo -e "\n############### Blocking Factor: $i ###############"
    echo -e "\n########## Time ##########"
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof ./hw3-2-$i $IN $OUT
    echo -e "\n########## Metric ##########"
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof -m $NVPROF_METRIC ./hw3-2-$i $IN $OUT
done

make clean