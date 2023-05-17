#!/bin/bash

seeds=(1111 2222 3333 4444 5555)
noise=(0.1)

for rng_seed in "${seeds[@]}"
do

  for prob in "${noise[@]}"
  do

    experiment_path=${TRACK_TO_LEARN_DATA}/experiments/
    tracking_folder=${experiment_path}/$1/$2/$rng_seed/test_scoring_${noise}_tractoinferno_1006_10

    data_path=${TRACK_TO_LEARN_DATA}/datasets/tractoinferno/sub-1006

    mkdir -p $tracking_folder

    python ttl_track.py \
      ${data_path}/fodf/sub-1006__fodf_6_descoteaux.nii.gz \
      ${data_path}/mask/sub-1006__mask_wm.nii.gz \
      ${data_path}/mask/sub-1006__mask_wm.nii.gz \
      ${data_path}/mask/sub-1006__mask_wm.nii.gz \
      ${data_path}/anat/sub-1006__T1w.nii.gz \
      ${experiment_path}/$1/$2/$rng_seed/model \
      ${experiment_path}/$1/$2/$rng_seed/model/hyperparameters.json \
      ${tracking_folder}/tractogram_${prob}_tractoinferno_1006_10.trk \
      --fa_map ${data_path}/dti/sub-1006__fa.nii.gz \
      --npv 10 --n_actor 25000 --compress 0.1 --prob $prob
  done
done
