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
max_ep=1000 # Chosen empirically
log_interval=50 # Log at n episodes

# Model params
prob=1.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=1 # Seed per voxel
theta=30 # Maximum angle for streamline curvature
# n_dirs=0

EXPERIMENT=SAC_Auto_FiberCupSearchOracle_v2

ID=$(date +"%F-%H_%M_%S")

rng_seed=1111

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

export COMET_OPTIMIZER_ID=99dde0cbe9004e10a9f54f7d5c85eabc

python TrackToLearn/searchers/sac_auto_searcher_oracle.py \
  $DEST_FOLDER \
  "$EXPERIMENT" \
  "$ID" \
  "${dataset_file}" \
  "${SUBJECT_ID}" \
  --rng_seed=${rng_seed} \
  --npv=${npv} \
  --theta=${theta} \
  --alignment_weighting=1.0 \
  --hidden_dims='1024-1024-1024' \
  --n_dirs=100 \
  --n_actor=${n_actor} \
  --action_type='cartesian' \
  --interface_seeding \
  --prob=${prob} \
  --use_gpu \
  --use_comet \
  --binary_stopping_threshold=0.1 \
  --oracle_validator \
  --oracle_stopping \
  --sparse_oracle_weighting=5.0 \
  --oracle_checkpoint='epoch_10_inferno.ckpt'


