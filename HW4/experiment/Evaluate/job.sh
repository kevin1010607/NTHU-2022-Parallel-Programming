#!/bin/bash
#SBATCH -J hw4-evaluate
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 12

DIR=./testcases
NODES=(2 4 3 4 4 4)
CPUS=(5 5 10 8 8 4)
JOB_NAME=(TEST01 TEST02 TEST03 TEST04 TEST05 TEST06)
NUM_REDUCER=(12 12 10 9 8 9)
DELAY=(4 1 3 4 4 1)
INPUT_FILE_NAME=(01.word 02.word 03.word 04.word 05.word 06.word)
CHUNK_SIZE=(2 10 4 10 20 20)
LOCALITY_CONFIG_FILENAME=(01.loc 02.loc 03.loc 04.loc 05.loc 06.loc)
OUTPUT_DIR=./output
ANSWER_FILE_NAME=(01.ans 02.ans 03.ans 04.ans 05.ans 06.ans)

# Compile code
make clean
make

mkdir -p $OUTPUT_DIR

for i in $(seq 0 5);
do
    mkdir -p $OUTPUT_DIR/${JOB_NAME[$i]}
    echo -e "\n############### JOB NAME: ${JOB_NAME[$i]} ###############"
    echo "srun -N${NODES[$i]} -c${CPUS[$i]} ./mapreduce ${JOB_NAME[$i]} ${NUM_REDUCER[$i]} ${DELAY[$i]} \
        $DIR/${INPUT_FILE_NAME[$i]} ${CHUNK_SIZE[$i]} $DIR/${LOCALITY_CONFIG_FILENAME[$i]} $OUTPUT_DIR/${JOB_NAME[$i]}"
    srun -N${NODES[$i]} -c${CPUS[$i]} ./mapreduce ${JOB_NAME[$i]} ${NUM_REDUCER[$i]} ${DELAY[$i]} \
        $DIR/${INPUT_FILE_NAME[$i]} ${CHUNK_SIZE[$i]} $DIR/${LOCALITY_CONFIG_FILENAME[$i]} $OUTPUT_DIR/${JOB_NAME[$i]}
    srun -N1 -n1 ./evaluate ${JOB_NAME[$i]} ${NUM_REDUCER[$i]} $DIR/${ANSWER_FILE_NAME[$i]} $OUTPUT_DIR/${JOB_NAME[$i]}
done

make clean