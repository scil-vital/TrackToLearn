#!/bin/bash
subjects=~/braindata/processedData/atheb/tracktolearn/datasets/tractoinferno
arr=(${subjects}/*)
arr=("${arr[@]##*/}")

set -e

export TRACK_TO_LEARN_DATA=~/braindata/processedData/atheb/tracktolearn

parallel -j3 ./scripts/rl_test_tractoinferno.sh SAC_Auto_ISMRM2015TrainOracle oracle_2023-11-13-15_09_01 {} ::: ${arr[@]} ::: 1111 2222 3333 4444 5555 ::: 1.0
