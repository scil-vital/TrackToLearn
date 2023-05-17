#!/usr/bin/env bash

set -e

# This should point to your dataset folder
DATASET_FOLDER=${TRACK_TO_LEARN_DATA}

# Data params
dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

n_actor=50000
npv=1
min_length=10
max_length=200

EXPERIMENT=$1
ID=$2

SEED=1111
SUBJECT_ID=hcp_100206
prob=0.2

EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"

dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

python ttl_validation.py \
  "$DEST_FOLDER" \
  "$EXPERIMENT" \
  "$ID" \
  "${dataset_file}" \
  "${SUBJECT_ID}" \
  "${reference_file}" \
  "${SCORING_DATA}" \
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

validation_folder=$DEST_FOLDER/tracking_"${prob}"_"${SUBJECT_ID}"

mkdir -p $validation_folder

mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/
