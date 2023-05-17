SUBJECT_ID=${@: -1}
DATASET_FOLDER=${TRACK_TO_LEARN_DATA}
dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5

# Env parameters
npv=2 # Seed per voxel
add_neighborhood=0.75 # Neighborhood to add to state input
theta=30 # Maximum angle for streamline curvature
min_length=20 # Minimum streamline length
max_length=200 # Maximum streamline length
step_size=0.75 # Step size (in mm)
n_signal=1 # Use last n input
n_dirs=4 # Also input last n directions taken

# Reward params
alignment_weighting=1. # Reward weighting for alignment
straightness_weighting=0. # Reward weighting for sinuosity
length_weighting=0.0 # Reward weighting for length
target_bonus_factor=0.0 # Reward penalty/bonus for end-of-trajectory actions
exclude_penalty_factor=0.0 # Reward penalty/bonus for end-of-trajectory actions
angle_penalty_factor=0.0 # Reward penalty/bonus for end-of-trajectory actions

rng_seed=1111

python ./scripts/reward_tractogram.py ${@:1:$#-1} \
  "${dataset_file}" \
  "${SUBJECT_ID}" \
  --rng_seed=${rng_seed} \
  --npv=${npv} \
  --theta=${theta} \
  --min_length=${min_length} \
  --max_length=${max_length} \
  --step_size=${step_size} \
  --add_neighborhood=${add_neighborhood} \
  --alignment_weighting=${alignment_weighting} \
  --straightness_weighting=${straightness_weighting} \
  --length_weighting=${length_weighting} \
  --target_bonus_factor=${target_bonus_factor} \
  --exclude_penalty_factor=${exclude_penalty_factor} \
  --angle_penalty_factor=${angle_penalty_factor} \
  --n_signal=${n_signal} \
  --n_dirs=${n_dirs}

