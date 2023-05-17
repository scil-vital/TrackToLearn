#!/usr/bin/env bash

set -e

# This should point to your dataset folder
DATASET_FOLDER=${TRACK_TO_LEARN_DATA}

# Should be relatively stable
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${SUBJECT_ID}/scoring_data

# Data params
dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

n_actor=10000
npv=7
min_length=20
max_length=200

EXPERIMENT=$1
ID=$2

validstds=(0.0 0.1)
subjectids=(ismrm2015)
seeds=(1111 2222 3333 4444 5555)

for SEED in "${seeds[@]}"
do
  for SUBJECT_ID in "${subjectids[@]}"
  do
    for prob in "${validstds[@]}"
    do
      EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
      SCORING_DATA=${DATASET_FOLDER}/datasets/${SUBJECT_ID}/scoring_data
      DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"

      dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
      reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

      echo $DEST_FOLDER/model/hyperparameters.json
      python ttl_validation.py \
        "$DEST_FOLDER" \
        "$EXPERIMENT" \
        "$ID" \
        "${dataset_file}" \
        "${SUBJECT_ID}" \
        "${reference_file}" \
        $DEST_FOLDER/model \
        $DEST_FOLDER/model/hyperparameters.json \
        --prob="${prob}" \
        --npv="${npv}" \
        --n_actor="${n_actor}" \
        --min_length="$min_length" \
        --max_length="$max_length" \
        --use_gpu \
        --fa_map="$DATASET_FOLDER"/datasets/${SUBJECT_ID}/dti/"${SUBJECT_ID}"_fa.nii.gz \
        --remove_invalid_streamlines

      validation_folder=$DEST_FOLDER/scoring_"${prob}"_"${SUBJECT_ID}"_${npv}

      mkdir -p $validation_folder

      mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/

      python scripts/score_tractogram.py \
        $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
        "$SCORING_DATA" \
        $validation_folder \
        --save_full_vc \
        --save_full_ic \
        --save_full_nc \
        --compute_ic_ib \
        --save_ib \
        --save_vb -f -v
    done
  done
done
