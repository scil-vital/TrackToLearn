#!/bin/bash

# Experiment name
# Example command: ./scripts/rl_experiment.sh TD3Experiment

set -e  # exit if any command fails

if [ -z "$1" ]; then
    echo "Missing experiment name"
fi

# This should point to your dataset folder
# DATASET_FOLDER= set in environment
# HOME_DATASET_FOLDER= set in environment

# BELOW SOME PARAMETERS THAT DEPEND ON MY FILE STRUCTURE
# YOU CAN CHANGE ANYTHING AS YOU WISH

# Should be relatively stable
TEST_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
HOME_EXPERIMENTS_FOLDER=${HOME_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID}/scoring_data

# Move stuff from data folder to working folder
mkdir -p $HOME_DATASET_FOLDER/raw/${SUBJECT_ID}

echo "Transfering data to working folder..."
cp -rn ${DATASET_FOLDER}/raw/${SUBJECT_ID} ${HOME_DATASET_FOLDER}/raw/
cp -rn ${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID} ${HOME_DATASET_FOLDER}/raw/

# Data params
dataset_file=$HOME_DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}_mask.hdf5
test_dataset_file=$HOME_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/${TEST_SUBJECT_ID}_mask.hdf5
test_reference_file=$HOME_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/masks/${TEST_SUBJECT_ID}_wm.nii.gz

# RL params
max_ep=150 # Chosen empirically
log_interval=500 # Log at n steps
lr=9.87e-6 # Learning rate
gamma=0.80 # Gamma for reward discounting
rng_seed=3533 # Seed for general randomness

# TD3 parameters
action_std=0.32 # STD deviation for action

# Env parameters
n_seeds_per_voxel=2 # Seed per voxel
max_angle=60 # Maximum angle for streamline curvature

EXPERIMENT=$1

ID=$(date +"%F-%H_%M_%S")

DEST_FOLDER=$HOME_EXPERIMENTS_FOLDER\"$EXPERIMENT"\"$ID"

python TrackToLearn/runners/td3_train.py \
  $DEST_FOLDER \
  $EXPERIMENT \
  $ID \
  ${dataset_file} \
  ${SUBJECT_ID} \
  ${test_dataset_file} \
  ${TEST_SUBJECT_ID} \
  ${test_reference_file} \
  ${SCORING_DATA} \
  --max_ep=${max_ep} \
  --log_interval=${log_interval} \
  --action_std=${action_std} \
  --lr=${lr} \
  --gamma=${gamma} \
  --rng_seed=${rng_seed} \
  --max_angle=${max_angle} \
  --use_gpu \
  --use_comet \
  --run_tractometer

n_seeds_per_voxel=33

validstds=(0.0 0.2 0.1 0.3)

for valid_noise in "${validstds[@]}"
do
  python TrackToLearn/runners/test.py $DEST_FOLDER \
    "$EXPERIMENT" \
    "$ID" \
    "${test_dataset_file}" \
    "${TEST_SUBJECT_ID}" \
    "${test_reference_file}" \
    "${SCORING_DATA}" \
    "$DEST_FOLDER/model" \
    "$DEST_FOLDER/model/hyperparameters.json" \
    --valid_noise="${valid_noise}" \
    --n_seeds_per_voxel="${n_seeds_per_voxel}" \
    --tracking_batch_size="${tracking_batch_size}" \
    --use_gpu \
    --remove_invalid_streamlines

  mkdir -p $DEST_FOLDER/scoring_"${valid_noise}"_fa

  python scripts/score_tractogram.py $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${TEST_SUBJECT_ID}".trk \
    $SCORING_DATA \
    $DEST_FOLDER/scoring_"${valid_noise}"_fa \
    --save_full_vc \
    --save_full_ic \
    --save_full_nc \
    --save_ib \
    --save_vb -f -v
done

mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/
