# Experiment name
# Example command: ./scripts/rl_experiment.sh TD3Experiment

set -e  # exit if any command fails

# This should point to your dataset folder
DATASET_FOLDER=${TRACK_TO_LEARN_DATA}

# Should be relatively stable
SUBJECT_ID=fibercup

# Create dataset
TrackToLearn/datasets/create_hdf5.py \
  $DATASET_FOLDER \
  ${SUBJECT_ID} \
  ${DATASET_FOLDER}/datasets/${SUBJECT_ID} \
  --name=${SUBJECT_ID}_mask \
  --fodfs \
  --add_masks
