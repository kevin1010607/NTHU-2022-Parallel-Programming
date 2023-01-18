#!/bin/bash
#SBATCH -J hw4-data_locality
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 12

DIR=./testcases
NODES=(4 4 4 4)
CPUS=(8 8 8 8)
JOB_NAME=(TEST05 TEST05 TEST05 TEST05)
NUM_REDUCER=(8 8 8 8)
DELAY=(4 4 4 4)
INPUT_FILE_NAME=(05.word 05.word 05.word 05.word)
CHUNK_SIZE=(20 20 20 20)
LOCALITY_CONFIG_FILENAME=(05_1.loc 05_2.loc 05_3.loc 05_4.loc)
OUTPUT_DIR=./output
ANSWER_FILE_NAME=(05.ans 05.ans 05.ans 05.ans)

# Compile code
make clean
make

mkdir -p $OUTPUT_DIR

for i in $(seq 0 3);
do
    mkdir -p $OUTPUT_DIR/${JOB_NAME[$i]}
    echo -e "\n############### Evenly distributed over number of nodes: $(expr $i + 1), Total number of nodes: 4 ###############"
    echo "srun -N${NODES[$i]} -c${CPUS[$i]} ./mapreduce ${JOB_NAME[$i]} ${NUM_REDUCER[$i]} ${DELAY[$i]} \
        $DIR/${INPUT_FILE_NAME[$i]} ${CHUNK_SIZE[$i]} $DIR/${LOCALITY_CONFIG_FILENAME[$i]} $OUTPUT_DIR/${JOB_NAME[$i]}"
    srun -N${NODES[$i]} -c${CPUS[$i]} ./mapreduce ${JOB_NAME[$i]} ${NUM_REDUCER[$i]} ${DELAY[$i]} \
        $DIR/${INPUT_FILE_NAME[$i]} ${CHUNK_SIZE[$i]} $DIR/${LOCALITY_CONFIG_FILENAME[$i]} $OUTPUT_DIR/${JOB_NAME[$i]}
    srun -N1 -n1 ./evaluate ${JOB_NAME[$i]} ${NUM_REDUCER[$i]} $DIR/${ANSWER_FILE_NAME[$i]} $OUTPUT_DIR/${JOB_NAME[$i]}
done

make clean