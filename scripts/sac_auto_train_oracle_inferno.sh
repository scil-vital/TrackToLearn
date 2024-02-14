#!/bin/bash

set -e  # exit if any command fails

DATASET_FOLDER=${TRACK_TO_LEARN_DATA}/
WORK_DATASET_FOLDER=${LOCAL_TRACK_TO_LEARN_DATA}/

SUBJECT_ID=tractoinferno
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments

mkdir -p $WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}

echo "Transfering data to working folder..."
rsync -rltv "${DATASET_FOLDER}"/datasets/${SUBJECT_ID} "${WORK_DATASET_FOLDER}"/datasets/

dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5

# RL params
max_ep=1000 # Chosen empirically
log_interval=50 # Log at n episodes

lr=0.0005 # Learning rate
gamma=0.95 # Gamma for reward discounting

# Model params
prob=1.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=2 # Seed per voxel
theta=30 # Maximum angle for streamline curvature
n_actor=4096

EXPERIMENT=SAC_Auto_InfernoTrainOracle

ID=$1_$(date +"%F-%H_%M_%S")

seeds=(1111 2222 3333 4444 5555)

for rng_seed in "${seeds[@]}"
do

  DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

  python -O TrackToLearn/trainers/sac_auto_train.py \
    $DEST_FOLDER \
    "$EXPERIMENT" \
    "$ID" \
    "${dataset_file}" \
    --max_ep=${max_ep} \
    --log_interval=${log_interval} \
    --lr=${lr} \
    --gamma=${gamma} \
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
    --coverage_weighting=0.0 \
    --oracle_validator \
    --oracle_stopping \
    --sparse_oracle_weighting=10.0 \
    --oracle_checkpoint='epoch_10_inferno.ckpt'

  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/
  cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

done
