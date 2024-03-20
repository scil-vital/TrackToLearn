#!/bin/bash

# Request resources --------------
# Graham GPU node: 12 cores, 10G ram, 1 GPU
#SBATCH --account $SALLOC_ACCOUNT
#SBATCH --gpus-per-node=v100:1    # Number of GPUs (per node)
#SBATCH --cpus-per-task=2         # Number of cores (not cpus)
#SBATCH --mem=12000M               # memory (per node)
#SBATCH --time=06-23:00            # time (DD-HH:MM)
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antoine.theberge@usherbrooke.ca

cd /home/$USER/projects/$SALLOC_ACCOUNT/$USER/TrackToLearn

module load python/3.10
pwd
source .env/bin/activate
module load httpproxy
export DISPLAY=:0

set -e  # exit if any command fails

# This should point to your dataset folder
HOME=~
WORK=$SLURM_TMPDIR
DATASET_FOLDER=${HOME}/projects/rrg-descotea/$USER/braindata/tracktolearn
WORK_DATASET_FOLDER=${WORK}/tracktolearn

VALIDATION_SUBJECT_ID=ismrm2015_nowm
SUBJECT_ID=ismrm2015_nowm
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
reference_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/anat/${VALIDATION_SUBJECT_ID}_T1.nii.gz

# RL params
max_ep=1000 # Chosen empirically
log_interval=50 # Log at n episodes

# Model params
prob=1.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=10 # Seed per voxel
theta=30 # Maximum angle for streamline curvature
# n_dirs=0

EXPERIMENT=SAC_Auto_ISMRM2015SearchOracle_Tractometer

ID=oracle_$(date +"%F-%H_%M_%S")

rng_seed=1111

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

python -O TrackToLearn/searchers/sac_auto_searcher_oracle.py \
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
  --hidden_dims='1024-1024-1024' \
  --n_dirs=100 \
  --n_actor=4096 \
  --action_type='cartesian' \
  --interface_seeding \
  --prob=${prob} \
  --use_gpu \
  --use_comet \
  --binary_stopping_threshold=0.1 \
  --tractometer_validator \
  --scoring_data=${SCORING_DATA} \
  --tractometer_weighting=10.0 \
  --oracle_validator \
  --sparse_oracle_weighting=0.0 \
  --oracle_checkpoint='epoch_39_ismrm2015v3.ckpt'

