#!/bin/bash

sub=$3
rng_seed=$4
prob=$5

experiment_path=${TRACK_TO_LEARN_DATA}/experiments/
tracking_folder=${TRACK_TO_LEARN_DATA}/experiments/$1/$2/tractoinferno_v2/testset/${sub}

data_path=${TRACK_TO_LEARN_DATA}/datasets/tractoinferno/${sub}
file=${tracking_folder}/tractogram_${prob}_tractoinferno_${sub}_20.trk

# if [[ -f $file ]]; then
#   echo $file "exists"
#   exit
# fi

echo "Tracking" ${sub} "seed" ${rng_seed}
mkdir -p $tracking_folder
ttl_track.py \
  ${data_path}/fodf/${sub}__fodf.nii.gz \
  ${data_path}/maps/${sub}__interface.nii.gz \
  ${data_path}/mask/${sub}__mask_wm.nii.gz \
  $file \
  --interface \
  --binary 0.1 \
  --agent ${experiment_path}/$1/$2/$rng_seed/model \
  --hyper ${experiment_path}/$1/$2/$rng_seed/model/hyperparameters.json \
  --npv 20 --n_actor 50000 --compress 0.2 --prob $prob -f
