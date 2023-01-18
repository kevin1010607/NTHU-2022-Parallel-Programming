#!/bin/bash
#SBATCH -J hw4-scalability
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 12

DIR=./testcases
NODES=(2 3 4)
CPUS=(4 4 4)
JOB_NAME=(TEST06 TEST06 TEST06)
NUM_REDUCER=(8 8 8)
DELAY=(2 2 2)
INPUT_FILE_NAME=(06.word 06.word 06.word)
CHUNK_SIZE=(20 20 20)
LOCALITY_CONFIG_FILENAME=(06.loc 06.loc 06.loc)
OUTPUT_DIR=./output
ANSWER_FILE_NAME=(06.ans 06.ans 06.ans)

# Compile code
make clean
make

mkdir -p $OUTPUT_DIR

for i in $(seq 0 2);
do
    mkdir -p $OUTPUT_DIR/${JOB_NAME[$i]}
    echo -e "\n############### Number of nodes: ${NODES[$i]} ###############"
    echo "srun -N${NODES[$i]} -c${CPUS[$i]} ./mapreduce ${JOB_NAME[$i]} ${NUM_REDUCER[$i]} ${DELAY[$i]} \
        $DIR/${INPUT_FILE_NAME[$i]} ${CHUNK_SIZE[$i]} $DIR/${LOCALITY_CONFIG_FILENAME[$i]} $OUTPUT_DIR/${JOB_NAME[$i]}"
    srun -N${NODES[$i]} -c${CPUS[$i]} ./mapreduce ${JOB_NAME[$i]} ${NUM_REDUCER[$i]} ${DELAY[$i]} \
        $DIR/${INPUT_FILE_NAME[$i]} ${CHUNK_SIZE[$i]} $DIR/${LOCALITY_CONFIG_FILENAME[$i]} $OUTPUT_DIR/${JOB_NAME[$i]}
    srun -N1 -n1 ./evaluate ${JOB_NAME[$i]} ${NUM_REDUCER[$i]} $DIR/${ANSWER_FILE_NAME[$i]} $OUTPUT_DIR/${JOB_NAME[$i]}
done

make clean