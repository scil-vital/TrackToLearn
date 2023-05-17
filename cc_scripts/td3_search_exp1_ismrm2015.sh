#!/bin/bash

# Request resources --------------
# Graham GPU node: 12 cores, 10G ram, 1 GPU
#SBATCH --account $SALLOC_ACCOUNT
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --cpus-per-task=12         # Number of cores (not cpus)
#SBATCH --mem=10000M               # memory (per node)
#SBATCH --time=06-23:00            # time (DD-HH:MM)
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

cd /home/$USER/projects/$SALLOC_ACCOUNT/$USER/TractoRL

module load python/3.8
pwd
source .env/bin/activate
module load httpproxy
export DISPLAY=:0

set -e  # exit if any command fails

# This should point to your dataset folder
HOME=~
WORK=$SLURM_TMPDIR
DATASET_FOLDER=${HOME}/projects/$SALLOC_ACCOUNT/$USER/tracktolearn
WORK_DATASET_FOLDER=${WORK}/tracktolearn
mkdir -p $WORK_DATASET_FOLDER

set -e  # exit if any command fails

VALIDATION_SUBJECT_ID=ismrm2015
SUBJECT_ID=ismrm2015
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${WORK_DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID}/scoring_data

# Move stuff from data folder to working folder
mkdir -p $WORK_DATASET_FOLDER/datasets

echo "Transfering data to working folder..."
cp -rn ${DATASET_FOLDER}/datasets/${SUBJECT_ID} ${WORK_DATASET_FOLDER}/datasets/
cp -rn ${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID} ${WORK_DATASET_FOLDER}/datasets/

# Data params
dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
validation_dataset_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/${VALIDATION_SUBJECT_ID}.hdf5
reference_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/masks/${VALIDATION_SUBJECT_ID}_wm.nii.gz
# RL params
max_ep=1000 # Chosen empirically
log_interval=50 # Log at n steps

# Model params
prob=0.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=2 # Seed per voxel
theta=60 # Maximum angle for streamline curvature
# n_dirs=0

EXPERIMENT=TD3_ISMRM2015SearchExp1

ID=$(date +"%F-%H_%M_%S")

rng_seed=1111

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

python TrackToLearn/searchers/td3_searcher.py \
  $DEST_FOLDER \
  "$EXPERIMENT" \
  "$ID" \
  "${dataset_file}" \
  "${SUBJECT_ID}" \
  "${validation_dataset_file}" \
  "${VALIDATION_SUBJECT_ID}" \
  "${reference_file}" \
  "${SCORING_DATA}" \
  --max_ep=${max_ep} \
  --log_interval=${log_interval} \
  --rng_seed=${rng_seed} \
  --npv=${npv} \
  --theta=${theta} \
  --prob=$prob \
  --use_gpu \
  --use_comet \
  --run_tractometer
  # --render

mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/"$rng_seed"
cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"
