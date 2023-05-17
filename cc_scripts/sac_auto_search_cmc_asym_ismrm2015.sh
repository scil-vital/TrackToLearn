#!/bin/bash

set -e  # exit if any command fails

DATASET_FOLDER=${TRACK_TO_LEARN_DATA}/
WORK_DATASET_FOLDER=~/${LOCAL_TRACK_TO_LEARN_DATA}/
mkdir -p $WORK_DATASET_FOLDER

VALIDATION_SUBJECT_ID=ismrm2015_asym
SUBJECT_ID=ismrm2015_asym
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID}/scoring_data

echo "Transfering data to working folder..."
mkdir -p $WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}
cp -rn $DATASET_FOLDER/datasets/${SUBJECT_ID} $WORK_DATASET_FOLDER/datasets/

dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
validation_dataset_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/${VALIDATION_SUBJECT_ID}.hdf5
reference_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/masks/${VALIDATION_SUBJECT_ID}_wm.nii.gz

max_ep=500 # Chosen empirically
log_interval=50 # Log at n steps

prob=0.0 # Noise to add to make a prob output. 0 for deterministic

npv=2 # Seed per voxel
theta=30 # Maximum angle for streamline curvature

EXPERIMENT=SACAutoISMRM2015Search_CmcAsym

ID=$(date +"%F-%H_%M_%S")_cmc_asym

seeds=(1111)

for rng_seed in "${seeds[@]}"
do

  DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

  python TrackToLearn/searchers/sac_auto_searcher.py \
    "$DEST_FOLDER" \
    "$EXPERIMENT" \
    "$ID" \
    "${dataset_file}" \
    "${SUBJECT_ID}" \
    "${validation_dataset_file}" \
    "${VALIDATION_SUBJECT_ID}" \
    "${reference_file}" \
    --scoring_data="${SCORING_DATA}" \
    --max_ep=${max_ep} \
    --log_interval=${log_interval} \
    --rng_seed=${rng_seed} \
    --npv=${npv} \
    --theta=${theta} \
    --prob=$prob \
    --use_gpu \
    --use_comet \
    --run_tractometer \
    --cmc \
    --asymmetric \
    --interface_seeding
    # --render

  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/
  cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

done
