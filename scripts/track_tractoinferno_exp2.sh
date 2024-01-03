#!/bin/bash
subjects=~/braindata/processedData/atheb/tracktolearn/datasets/tractoinferno
arr=(${subjects}/*)
arr=("${arr[@]##*/}")

export TRACK_TO_LEARN_DATA=~/braindata/processedData/atheb/tracktolearn

# parallel -j2 ./scripts/rl_test_tractoinferno_exp2.sh VPG_ISMRM2015TrainExp2 2023-02-14-11_19_01 {} ::: ${arr[@]} ::: 1111 2222 3333 4444 5555 ::: 0.0
# parallel -j2 ./scripts/rl_test_tractoinferno_exp2.sh A2C_ISMRM2015TrainExp2 2023-02-14-11_19_01 {} ::: ${arr[@]} ::: 1111 2222 3333 4444 5555 ::: 0.0
# parallel -j2 ./scripts/rl_test_tractoinferno_exp2.sh ACKTR_ISMRM2015TrainExp2 2023-02-21-09_58_36 {} ::: ${arr[@]} ::: 1111 2222 3333 4444 5555 ::: 0.0
# parallel -j2 ./scripts/rl_test_tractoinferno_exp2.sh TRPO_ISMRM2015TrainExp2 2023-02-14-13_11_18 {} ::: ${arr[@]} ::: 1111 2222 3333 4444 5555 ::: 0.0
# parallel -j2 ./scripts/rl_test_tractoinferno_exp2.sh PPO_ISMRM2015TrainExp2 2023-03-06-12_22_46 {} ::: ${arr[@]} ::: 1111 2222 3333 4444 5555 ::: 0.0
# parallel -j2 ./scripts/rl_test_tractoinferno_exp2.sh DDPG_ISMRM2015TrainExp2 2023-02-15-04_15_36 {} ::: ${arr[@]} ::: 1111 2222 3333 4444 5555 ::: 0.1
# parallel -j2 ./scripts/rl_test_tractoinferno_exp2.sh TD3_ISMRM2015TrainExp2 2023-02-15-12_59_38 {} ::: ${arr[@]} ::: 1111 2222 3333 4444 5555 ::: 0.1
# parallel -j2 ./scripts/rl_test_tractoinferno_exp2.sh SAC_ISMRM2015TrainExp2 2023-02-16-10_04_08 {} ::: ${arr[@]} ::: 1111 2222 3333 4444 5555 ::: 0.1
parallel -j1 ./scripts/rl_test_tractoinferno_exp2.sh SAC_Auto_ISMRM2015TrainExp2 2023-02-21-17_27_47 {} ::: sub-1046 ::: 4444 5555 ::: 0.1
