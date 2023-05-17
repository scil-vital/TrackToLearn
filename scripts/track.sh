#!/usr/bin/env bash

set -e

# This should point to your dataset folder

n_actor=10000
npv=10
min_length=20
max_length=200

EXPERIMENT=$1
ID=$2

# SEED=1111
SUBJECT_ID=hcp_100206
prob=0.1

DATASET_FOLDER=${TRACK_TO_LEARN_DATA}

reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/anat/${SUBJECT_ID}_t1.nii.gz
signal_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/hcp_100206_signal.nii.gz
peaks_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/fodfs/hcp_100206_peaks.nii.gz
in_seed=$DATASET_FOLDER/datasets/${SUBJECT_ID}/maps/hcp_100206_interface.nii.gz
in_mask=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/hcp_100206_wm.nii.gz
target_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/hcp_100206_gm.nii.gz
exclude_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/hcp_100206_csf.nii.gz

seeds=(1111 2222 3333 4444 5555)
for SEED in "${seeds[@]}"
do

  EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
  DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"
  out_tractogram="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk

  python ttl_track.py \
    "$DEST_FOLDER" \
    "$EXPERIMENT" \
    "$ID" \
    "${signal_file}" \
    "${peaks_file}" \
    "${in_seed}" \
    "${in_mask}" \
    "${target_file}" \
    "${exclude_file}" \
    "${SUBJECT_ID}" \
    "${reference_file}" \
    $DEST_FOLDER/model \
    $DEST_FOLDER/model/hyperparameters.json \
    --out_tractogram="${out_tractogram}" \
    --prob="${prob}" \
    --npv="${npv}" \
    --n_actor="${n_actor}" \
    --min_length="$min_length" \
    --max_length="$max_length" \
    --use_gpu \
    --fa_map="$DATASET_FOLDER"/datasets/${SUBJECT_ID}/dti/"${SUBJECT_ID}"_fa.nii.gz \
    --interface_seeding \
    --remove_invalid_streamlines

  validation_folder=$DEST_FOLDER/tracking_"${prob}"_"${SUBJECT_ID}"

  mkdir -p $validation_folder

  mv $out_tractogram $validation_folder/
done

