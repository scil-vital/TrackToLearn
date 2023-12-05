#!/bin/bash

set -e

# seeds=(1111 2222 3333 4444 5555)
prob=$5

sub=$3
rng_seed=$4

# for rng_seed in "${seeds[@]}"
# do
experiment_path=${TRACK_TO_LEARN_DATA}/experiments/
tracking_folder=${TRACK_TO_LEARN_DATA}/experiments/$1/$2/$rng_seed/tractoinferno_test/${sub}/

data_path=${TRACK_TO_LEARN_DATA}/datasets/tractoinferno/${sub}
file=${tracking_folder}/tractogram_${prob}_tractoinferno_${sub}_10.trk

if [[ -f $file ]]; then
  echo $file "exists"
  exit
fi

echo "Tracking" ${sub} "seed" ${rng_seed}
mkdir -p $tracking_folder

ttl_track.py \
  ${data_path}/fodf/${sub}__fodf.nii.gz \
  ${data_path}/mask/${sub}__mask_wm.nii.gz \
  ${data_path}/mask/${sub}__mask_wm.nii.gz \
  $file \
  --policy ${experiment_path}/$1/$2/$rng_seed/model \
  --hyper ${experiment_path}/$1/$2/$rng_seed/model/hyperparameters.json \
  --fa_map ${data_path}/dti/${sub}__fa.nii.gz \
  --npv 10 --n_actor 50000 --compress 0.2 --prob $prob -f
# done
