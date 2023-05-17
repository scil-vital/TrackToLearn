#!/bin/bash

# Request resources --------------
# Graham GPU node: 12 cores, 10G ram, 1 GPU
#SBATCH --account $SALLOC_ACCOUNT
#SBATCH --gpus-per-node=v100l:1    # Number of GPUs (per node)
#SBATCH --cpus-per-task=12         # Number of cores (not cpus)
#SBATCH --mem=12000M               # memory (per node)
#SBATCH --time=01-00:00            # time (DD-HH:MM)
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-4                # IMPORTANT

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

VALIDATION_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID}/scoring_data

# Move stuff from data folder to working folder
mkdir -p $WORK_DATASET_FOLDER/datasets

echo "Transfering data to working folder..."
cp -rn ${DATASET_FOLDER}/datasets/${SUBJECT_ID} ${WORK_DATASET_FOLDER}/datasets/
cp -rn ${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID} ${WORK_DATASET_FOLDER}/datasets/

dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
validation_dataset_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/${VALIDATION_SUBJECT_ID}.hdf5
reference_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/masks/${VALIDATION_SUBJECT_ID}_wm.nii.gz

# RL params
max_ep=1000 # Chosen empirically
log_interval=50 # Log at n episodes
lr=0.00001 # Learning rate
gamma=0.5 # Gamma for reward discounting

# Model params
prob=0.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=10 # Seed per voxel
theta=30 # Maximum angle for streamline curvature

EXPERIMENT=SAC_Auto_FiberCupTrainLength1Exp5

ID=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

seeds=(1111 2222 3333 4444 5555)

rng_seed=${seeds[${SLURM_ARRAY_TASK_ID}]}

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

python TrackToLearn/runners/sac_auto_train.py \
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
  --lr=${lr} \
  --gamma=${gamma} \
  --rng_seed=${rng_seed} \
  --npv=${npv} \
  --theta=${theta} \
  --length_weighting=1 \
  --use_gpu \
  --use_comet \
  --run_tractometer

mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/
cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

# done
