#!/bin/bash

seeds=(1111 2222 3333 4444 5555)
noise=(0.1)

for rng_seed in "${seeds[@]}"
do

  for prob in "${noise[@]}"
  do

    experiment_path=${TRACK_TO_LEARN_DATA}/experiments/

    data_path=${TRACK_TO_LEARN_DATA}/datasets/tractoinferno/
    for sub in $data_path/; do
      s=$(basename $sub)
      sub_path=${TRACK_TO_LEARN_DATA}/datasets/tractoinferno/${s}
      tracking_folder=${experiment_path}/$1/$2/$rng_seed/test_scoring_${noise}_tractoinferno_${s}_10

      mkdir -p $tracking_folder

      ttl_track.py \
        ${sub_path}/fodf/sub-1006__fodf_6_descoteaux.nii.gz \
        ${sub_path}/mask/sub-1006__mask_wm.nii.gz \
        ${sub_path}/mask/sub-1006__mask_wm.nii.gz \
        ${sub_path}/mask/sub-1006__mask_wm.nii.gz \
        ${sub_path}/anat/sub-1006__T1w.nii.gz \
        ${experiment_path}/$1/$2/$rng_seed/model \
        ${experiment_path}/$1/$2/$rng_seed/model/hyperparameters.json \
        ${tracking_folder}/tractogram_${prob}_tractoinferno_${s}_10.trk \
        --fa_map ${data_path}/dti/sub-1006__fa.nii.gz \
        --npv 10 --n_actor 25000 --compress 0.1 --prob $prob
      done
  done
done
