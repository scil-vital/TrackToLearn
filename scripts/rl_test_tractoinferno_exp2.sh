#!/bin/bash

# seeds=(1111 2222 3333 4444 5555)


sub=$3
rng_seed=$4
prob=$5

# for rng_seed in "${seeds[@]}"
# do

experiment_path=${TRACK_TO_LEARN_DATA}/experiments/

data_path=${TRACK_TO_LEARN_DATA}/datasets/tractoinferno/${sub}
file=$6

# if [[ -f $file ]]; then
#   echo $file "exists"
#   exit
# fi

echo "Tracking" ${sub} "seed" ${rng_seed}
ttl_track.py \
  ${data_path}/fodf/${sub}__fodf.nii.gz \
  ${data_path}/maps/${sub}__interface.nii.gz \
  ${data_path}/mask/${sub}__mask_wm.nii.gz \
  $file \
  --interface --binary=0.1 \
  --agent ${experiment_path}/$1/$2/$rng_seed/model \
  --hyper ${experiment_path}/$1/$2/$rng_seed/model/hyperparameters.json \
  --npv 20 --n_actor 100000 --compress 0.2 --prob $prob -f
