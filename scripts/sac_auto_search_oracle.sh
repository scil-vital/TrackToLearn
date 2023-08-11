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
max_ep=750 # Chosen empirically
log_interval=50 # Log at n episodes

# Model params
prob=0.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=100 # Seed per voxel
theta=30 # Maximum angle for streamline curvature
# n_dirs=0

EXPERIMENT=SAC_Auto_FiberCupSearchOracle_v2

ID=$(date +"%F-%H_%M_%S")

rng_seed=1111

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

export COMET_OPTIMIZER_ID=1dc20c2b8854402080e58a231cfd85b6

python TrackToLearn/searchers/sac_auto_searcher_oracle.py \
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
  --n_dirs=100 \
  --action_type='cartesian' \
  --interface_seeding \
  --use_gpu \
  --use_comet \
  --run_oracle='epoch_49_fibercup_transformer.ckpt' \
  --run_tractometer=${SCORING_DATA}
