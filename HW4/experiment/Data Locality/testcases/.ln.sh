#!/bin/sh
DIR=/home/pp22/share/.testcases/hw4

for dir in $DIR/*_sample_ans; do
        echo $dir
        echo $(basename "${dir}")
        ln -s $dir $(basename "${dir}")
done

