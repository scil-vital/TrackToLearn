#!/bin/bash

set -e

npv=100

experiment=$1
id=$2

sub=$3
rng_seed=$4

experiment_path=${TRACK_TO_LEARN_DATA}/experiments/
tracking_folder=${TRACK_TO_LEARN_DATA}/experiments/$experiment/$id/${sub}

data_path=${TRACK_TO_LEARN_DATA}/datasets/${sub}
file=${tracking_folder}/tractogram_${sub}_${rng_seed}_${npv}.trk

echo "Tracking" ${sub} "seed" ${rng_seed}
mkdir -p $tracking_folder

ttl_track.py \
  ${data_path}/fodfs/${sub}__fodf.nii.gz \
  ${data_path}/maps/${sub}__interface.nii.gz \
  ${data_path}/masks/${sub}__wm.nii.gz \
  $file \
  --agent ${experiment_path}/$1/$2/$rng_seed/model \
  --hyper ${experiment_path}/$1/$2/$rng_seed/model/hyperparameters.json \
  --binary=0.1 \
  --min_length=20 \
  --max_length=200 \
  --npv ${npv} --n_actor 25000 --compress 0.2 -f
# done
