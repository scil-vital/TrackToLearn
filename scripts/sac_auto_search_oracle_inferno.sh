#!/bin/bash

set -e  # exit if any command fails

DATASET_FOLDER=${TRACK_TO_LEARN_DATA}/
WORK_DATASET_FOLDER=${LOCAL_TRACK_TO_LEARN_DATA}/

SUBJECT_ID=tractoinferno
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID}/scoring_data

mkdir -p $WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}

echo "Transfering data to working folder..."
rsync -rltv "${DATASET_FOLDER}"/datasets/${SUBJECT_ID} "${WORK_DATASET_FOLDER}"/datasets/

dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5

# RL params
max_ep=1000 # Chosen empirically
log_interval=50 # Log at n episodes

# Model params
lr=0.0005 # Learning rate
prob=1.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=10 # Seed per voxel
theta=30 # Maximum angle for streamline curvature
# n_dirs=0

EXPERIMENT=SAC_Auto_InfernoSearchOracle

ID=oracle_$(date +"%F-%H_%M_%S")

rng_seed=1111

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

python -O TrackToLearn/searchers/sac_auto_searcher_oracle.py \
  $DEST_FOLDER \
  "$EXPERIMENT" \
  "$ID" \
  "${dataset_file}" \
  --max_ep=${max_ep} \
  --log_interval=${log_interval} \
  --rng_seed=${rng_seed} \
  --npv=${npv} \
  --theta=${theta} \
  --lr=${lr} \
  --alignment_weighting=1.0 \
  --hidden_dims='1024-1024-1024' \
  --n_dirs=100 \
  --n_actor=4096 \
  --action_type='cartesian' \
  --interface_seeding \
  --prob=${prob} \
  --use_gpu \
  --use_comet \
  --binary_stopping_threshold=0.1 \
  --coverage_weighting=0.0 \
  --oracle_validator \
  --oracle_stopping \
  --oracle_checkpoint='epoch_10_inferno.ckpt'
