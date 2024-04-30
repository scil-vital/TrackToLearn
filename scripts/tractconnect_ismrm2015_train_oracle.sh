#!/bin/bash

set -e  # exit if any command fails

DATASET_FOLDER=${TRACK_TO_LEARN_DATA}/
WORK_DATASET_FOLDER=${LOCAL_TRACK_TO_LEARN_DATA}/

SUBJECT_ID=ismrm2015
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments

mkdir -p $WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}

echo "Transfering data to working folder..."
rsync -rltv "${DATASET_FOLDER}"/datasets/${SUBJECT_ID} "${WORK_DATASET_FOLDER}"/datasets/

dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}_freesurfer.hdf5
scoring_data=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/scoring_data
reference=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/anat/${SUBJECT_ID}__t1.nii.gz

# RL params
max_ep=1000 # Chosen empirically
log_interval=50 # Log at n episodes

lr=0.0005 # Learning rate
gamma=0.99 # Gamma for reward discounting

# Env parameters
npv=10 # Seed per voxel
theta=30 # Maximum angle for streamline curvature
step=0.5

EXPERIMENT=OracleISMRM2015

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
    --lr=${lr} \
    --gamma=${gamma} \
    --npv=${npv} \
    --theta=${theta} \
    --step=${step} \
    --n_dirs=100 \
    --connectivity_bonus=10 \
    --connectivity_validator \
    --oracle_checkpoint="models/epoch_49_ismrm2015v4.ckpt" \
    --oracle_bonus=10 \
    --oracle_stopping \
    --tractometer_validator \
    --tractometer_dilate=1 \
    --scoring_data=${scoring_data} \
    --tractometer_reference=${reference} \
    --log_interval=${log_interval} \
    --rng_seed=${rng_seed} \
    --max_ep=${max_ep} \
    --use_comet

  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/
  cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

done
