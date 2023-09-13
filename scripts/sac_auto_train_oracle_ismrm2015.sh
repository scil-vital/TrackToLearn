#!/bin/bash

set -e  # exit if any command fails

DATASET_FOLDER=${TRACK_TO_LEARN_DATA}/
WORK_DATASET_FOLDER=${LOCAL_TRACK_TO_LEARN_DATA}/

VALIDATION_SUBJECT_ID=ismrm2015
SUBJECT_ID=ismrm2015
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID}/scoring_data

mkdir -p $WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}

echo "Transfering data to working folder..."
cp -rnv "${DATASET_FOLDER}"/datasets/${VALIDATION_SUBJECT_ID} "${WORK_DATASET_FOLDER}"/datasets/
cp -rnv "${DATASET_FOLDER}"/datasets/${SUBJECT_ID} "${WORK_DATASET_FOLDER}"/datasets/

dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
validation_dataset_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/${VALIDATION_SUBJECT_ID}.hdf5
reference_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/anat/${VALIDATION_SUBJECT_ID}_T1.nii.gz

# RL params
max_ep=1000 # Chosen empirically
log_interval=50 # Log at n episodes

lr=0.0005 # Learning rate
gamma=0.99 # Gamma for reward discounting

# Model params
prob=0.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=10 # Seed per voxel
theta=30 # Maximum angle for streamline curvature
n_actor=10000

EXPERIMENT=SAC_Auto_ISMRM2015TrainOracle

ID=$(date +"%F-%H_%M_%S")

seeds=(1111 2222 3333 4444 5555)

for rng_seed in "${seeds[@]}"
do

  DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

  python -O TrackToLearn/trainers/sac_auto_train.py \
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
    --lr=${lr} \
    --gamma=${gamma} \
    --rng_seed=${rng_seed} \
    --npv=${npv} \
    --theta=${theta} \
    --alignment_weighting=1.0 \
    --oracle_weighting=0.0 \
    --hidden_dims='1024-1024-1024' \
    --n_dirs=100 \
    --n_actor=${n_actor} \
    --action_type='cartesian' \
    --interface_seeding \
    --use_gpu \
    --use_comet \
    --run_oracle='feedforward_epoch99_ismrm2015.ckpt' \
    --run_tractometer=${SCORING_DATA}

  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/
  cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

done
