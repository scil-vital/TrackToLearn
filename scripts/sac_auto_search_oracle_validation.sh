#!/bin/bash

set -e  # exit if any command fails

DATASET_FOLDER=${TRACK_TO_LEARN_DATA}/
WORK_DATASET_FOLDER=${LOCAL_TRACK_TO_LEARN_DATA}/

VALIDATION_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID}/scoring_data

mkdir -p $WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}

echo "Transfering data to working folder..."
cp -rnv "${DATASET_FOLDER}"/datasets/${VALIDATION_SUBJECT_ID} "${WORK_DATASET_FOLDER}"/datasets/
cp -rnv "${DATASET_FOLDER}"/datasets/${SUBJECT_ID} "${WORK_DATASET_FOLDER}"/datasets/

dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
validation_dataset_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/${VALIDATION_SUBJECT_ID}.hdf5
reference_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/masks/${VALIDATION_SUBJECT_ID}_wm.nii.gz

# RL params
max_ep=500 # Chosen empirically
log_interval=50 # Log at n episodes

# Model params
prob=0.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=100 # Seed per voxel
theta=30 # Maximum angle for streamline curvature
# n_dirs=0

EXPERIMENT=SAC_Auto_FiberCupSearchOracleValidation

ID=$(date +"%F-%H_%M_%S")

rng_seed=1111

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

export COMET_OPTIMIZER_ID=8bb26d0855c84c4fb1d4e876f3a84eb0

python TrackToLearn/searchers/sac_auto_searcher.py \
  $DEST_FOLDER \
  "$EXPERIMENT" \
  "$ID" \
  "${dataset_file}" \
  "${SUBJECT_ID}" \
  "${validation_dataset_file}" \
  "${VALIDATION_SUBJECT_ID}" \
  "${reference_file}" \
  --max_ep=${max_ep} \
  --log_interval=${log_interval} \
  --rng_seed=${rng_seed} \
  --npv=${npv} \
  --theta=${theta} \
  --alignment_weighting=1.0 \
  --oracle_weighting=0.0 \
  --coverage_weighting=0.0 \
  --n_dirs=4 \
  --action_type='cartesian' \
  --interface_seeding \
  --use_gpu \
  --use_comet \
  --run_oracle='checkpoint.ckpt' \
  --run_tractometer=${SCORING_DATA}
