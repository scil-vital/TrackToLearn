
# This should point to your dataset folder
DATASET_FOLDER=${TRACK_TO_LEARN_DATA}

# Should be relatively stable
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments

step_size=0.75 # Step size (in mm)

npv=33
min_length=20
max_length=200

STEP=0.75
MIN=20
MAX=200
seeds=(1111 2222 3333 4444 5555)

# SUBJECT_ID=fibercup
# EXPERIMENT=PFT_FiberCupExp1
# npv=33
# 
# ID=$(date +"%F-%H_%M_%S")
# 
# for seed in "${seeds[@]}";
# do
# 
#   OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk
# 
#   BASE=${DATASET_FOLDER}/datasets/${SUBJECT_ID}
#   SCORING_DATA=${BASE}/scoring_data
#   DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
#   mkdir -p $DEST_FOLDER
# 
#   scil_compute_pft.py \
#     $BASE/fodfs/${SUBJECT_ID}_fodf.nii.gz \
#     $BASE/masks/${SUBJECT_ID}_wm.nii.gz \
#     $BASE/maps/map_include.nii.gz \
#     $BASE/maps/map_exclude.nii.gz  \
#     $DEST_FOLDER/$OUT \
#     --npv $npv --min_length $MIN \
#     --max_length $MAX --step $STEP \
#     --seed $seed -f -v
# 
#   validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}
# 
#   mkdir -p $validation_folder
# 
#   mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/
# 
#   python scripts/score_tractogram.py \
#     $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
#     "$SCORING_DATA" \
#     $validation_folder \
#     --compute_ic_ib \
#     --save_full_vc \
#     --save_full_ic \
#     --save_full_nc \
#     --save_ib \
#     --save_vb -f -v
# 
# done
# 
# EXPERIMENT=PFT_FiberCupExp2
# npv=300
# 
# ID=$(date +"%F-%H_%M_%S")
# 
# for seed in "${seeds[@]}";
# do
# 
#   OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk
# 
#   BASE=${DATASET_FOLDER}/datasets/${SUBJECT_ID}
#   SCORING_DATA=${BASE}/scoring_data
#   DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
#   mkdir -p $DEST_FOLDER
# 
#   scil_compute_pft.py \
#     $BASE/fodfs/${SUBJECT_ID}_fodf.nii.gz \
#     $BASE/maps/interface.nii.gz \
#     $BASE/maps/map_include.nii.gz \
#     $BASE/maps/map_exclude.nii.gz  \
#     $DEST_FOLDER/$OUT \
#     --npv $npv --min_length $MIN \
#     --max_length $MAX --step $STEP \
#     --seed $seed -f -v
# 
#   validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}
# 
#   mkdir -p $validation_folder
# 
#   mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/
# 
#   python scripts/score_tractogram.py \
#     $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
#     "$SCORING_DATA" \
#     $validation_folder \
#     --compute_ic_ib \
#     --save_full_vc \
#     --save_full_ic \
#     --save_full_nc \
#     --save_ib \
#     --save_vb -f -v
# 
# done
# 
# SUBJECT_ID=fibercup_flipped
# EXPERIMENT=PFT_FiberCupExp1
# BASE=${TRACK_TO_LEARN_DATA}/datasets/$SUBJECT_ID
# npv=33
# 
# ID=$(date +"%F-%H_%M_%S")
# 
# for seed in "${seeds[@]}";
# do
# 
#   OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk
# 
#   BASE=${DATASET_FOLDER}/datasets/${SUBJECT_ID}
#   SCORING_DATA=${BASE}/scoring_data
#   DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
#   mkdir -p $DEST_FOLDER
# 
#   scil_compute_pft.py \
#     $BASE/fodfs/${SUBJECT_ID}_fodf.nii.gz \
#     $BASE/masks/${SUBJECT_ID}_wm.nii.gz \
#     $BASE/maps/map_include.nii.gz \
#     $BASE/maps/map_exclude.nii.gz  \
#     $DEST_FOLDER/$OUT \
#     --npv $npv --min_length $MIN \
#     --max_length $MAX --step $STEP \
#     --seed $seed -f -v
# 
#   validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}
# 
#   mkdir -p $validation_folder
# 
#   mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/
# 
#   python scripts/score_tractogram.py \
#     $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
#     "$SCORING_DATA" \
#     $validation_folder \
#     --compute_ic_ib \
#     --save_full_vc \
#     --save_full_ic \
#     --save_full_nc \
#     --save_ib \
#     --save_vb -f -v
# 
# done
# 
# SUBJECT_ID=fibercup_flipped
# EXPERIMENT=PFT_FiberCupExp2
# npv=300
# 
# ID=$(date +"%F-%H_%M_%S")
# 
# for seed in "${seeds[@]}";
# do
# 
#   OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk
# 
#   BASE=${DATASET_FOLDER}/datasets/${SUBJECT_ID}
#   SCORING_DATA=${BASE}/scoring_data
#   DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
#   mkdir -p $DEST_FOLDER
# 
#   scil_compute_pft.py \
#     $BASE/fodfs/${SUBJECT_ID}_fodf.nii.gz \
#     $BASE/maps/interface.nii.gz \
#     $BASE/maps/map_include.nii.gz \
#     $BASE/maps/map_exclude.nii.gz  \
#     $DEST_FOLDER/$OUT \
#     --npv $npv --min_length $MIN \
#     --max_length $MAX --step $STEP \
#     --seed $seed -f -v
# 
#   validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}
# 
#   mkdir -p $validation_folder
# 
#   mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/
# 
#   python scripts/score_tractogram.py \
#     $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
#     "$SCORING_DATA" \
#     $validation_folder \
#     --compute_ic_ib \
#     --save_full_vc \
#     --save_full_ic \
#     --save_full_nc \
#     --save_ib \
#     --save_vb -f -v
# 
# done
# 
# SUBJECT_ID=ismrm2015
# EXPERIMENT=PFT_ISMRM2015Exp1
# BASE=${TRACK_TO_LEARN_DATA}/datasets/$SUBJECT_ID
# npv=7
# 
# ID=$(date +"%F-%H_%M_%S")
# 
# for seed in "${seeds[@]}";
# do
# 
#   OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk
# 
#   BASE=${DATASET_FOLDER}/datasets/${SUBJECT_ID}
#   SCORING_DATA=${BASE}/scoring_data
#   DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
#   mkdir -p $DEST_FOLDER
# 
#   scil_compute_pft.py \
#     $BASE/fodfs/${SUBJECT_ID}_fodf.nii.gz \
#     $BASE/masks/${SUBJECT_ID}_wm.nii.gz \
#     $BASE/maps/map_include.nii.gz \
#     $BASE/maps/map_exclude.nii.gz  \
#     $DEST_FOLDER/$OUT \
#     --npv $npv --min_length $MIN \
#     --max_length $MAX --step $STEP \
#     --seed $seed -f -v
# 
#   validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}
# 
#   mkdir -p $validation_folder
# 
#   mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/
# 
#   python scripts/score_tractogram.py \
#     $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
#     "$SCORING_DATA" \
#     $validation_folder \
#     --compute_ic_ib \
#     --save_full_vc \
#     --save_full_ic \
#     --save_full_nc \
#     --save_ib \
#     --save_vb -f -v
# 
# done
# 
# SUBJECT_ID=ismrm2015
# EXPERIMENT=PFT_ISMRM2015Exp2
# npv=60
# 
# ID=$(date +"%F-%H_%M_%S")
# 
# for seed in "${seeds[@]}";
# do
# 
#   OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk
# 
#   BASE=${DATASET_FOLDER}/datasets/${SUBJECT_ID}
#   SCORING_DATA=${BASE}/scoring_data
#   DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
#   mkdir -p $DEST_FOLDER
# 
#   scil_compute_pft.py \
#     $BASE/fodfs/${SUBJECT_ID}_fodf.nii.gz \
#     $BASE/maps/interface.nii.gz \
#     $BASE/maps/map_include.nii.gz \
#     $BASE/maps/map_exclude.nii.gz  \
#     $DEST_FOLDER/$OUT \
#     --npv $npv --min_length $MIN \
#     --max_length $MAX --step $STEP \
#     --seed $seed -f -v
# 
#   validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}
# 
#   mkdir -p $validation_folder
# 
#   mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/
# 
#   python scripts/score_tractogram.py \
#     $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
#     "$SCORING_DATA" \
#     $validation_folder \
#     --compute_ic_ib \
#     --save_full_vc \
#     --save_full_ic \
#     --save_full_nc \
#     --save_ib \
#     --save_vb -f -v
# 
# done
# 
# STEP=0.375
# SUBJECT_ID=hcp_100206
# EXPERIMENT=PFT_ISMRM2015Exp1
# BASE=${TRACK_TO_LEARN_DATA}/datasets/$SUBJECT_ID
# npv=2
# 
# ID=2023-02-24-17_45_03
# 
# for seed in "${seeds[@]}";
# do
# 
#   OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk
# 
#   BASE=${DATASET_FOLDER}/datasets/${SUBJECT_ID}
#   SCORING_DATA=${BASE}/scoring_data
#   DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
#   mkdir -p $DEST_FOLDER
# 
#   scil_compute_pft.py \
#     $BASE/fodfs/${SUBJECT_ID}_fodf.nii.gz \
#     $BASE/masks/${SUBJECT_ID}_wm.nii.gz \
#     $BASE/maps/${SUBJECT_ID}_map_include.nii.gz \
#     $BASE/maps/${SUBJECT_ID}_map_exclude.nii.gz  \
#     $DEST_FOLDER/$OUT \
#     --npv $npv --min_length $MIN \
#     --max_length $MAX --step $STEP \
#     --seed $seed -f -v
# 
#   validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}
# 
#   mkdir -p $validation_folder
# 
#   mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/
# 
#   scil_recognize_multi_bundles.py \
#     $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
#     $SCORING_DATA/config/config_ind.json \
#     $SCORING_DATA/atlas/*/ \
#     $SCORING_DATA/output0GenericAffine.mat \
#     --out $validation_folder/voting_results \
#     -f --log_level DEBUG --multi_parameters 27 \
#     --minimal_vote 0.4 --tractogram_clustering 8 10 12 \
#     --processes 4 --seeds 0
# 
# done
# 
# SUBJECT_ID=hcp_100206
# EXPERIMENT=PFT_ISMRM2015Exp2
# npv=10
# 
# ID=2023-02-24-21_08_25
# 
# for seed in "${seeds[@]}";
# do
# 
#   OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk
# 
#   BASE=${DATASET_FOLDER}/datasets/${SUBJECT_ID}
#   SCORING_DATA=${BASE}/scoring_data
#   DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
#   mkdir -p $DEST_FOLDER
# 
#   scil_compute_pft.py \
#     $BASE/fodfs/${SUBJECT_ID}_fodf.nii.gz \
#     $BASE/maps/${SUBJECT_ID}_interface.nii.gz \
#     $BASE/maps/${SUBJECT_ID}_map_include.nii.gz \
#     $BASE/maps/${SUBJECT_ID}_map_exclude.nii.gz  \
#     $DEST_FOLDER/$OUT \
#     --npv $npv --min_length $MIN \
#     --max_length $MAX --step $STEP \
#     --seed $seed -f -v
# 
#   validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}
# 
#   mkdir -p $validation_folder
# 
#   mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/
# 
#   scil_recognize_multi_bundles.py \
#     $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
#     $SCORING_DATA/config/config_ind.json \
#     $SCORING_DATA/atlas/*/ \
#     $SCORING_DATA/output0GenericAffine.mat \
#     --out $validation_folder/voting_results \
#     -f --log_level DEBUG --multi_parameters 27 \
#     --minimal_vote 0.4 --tractogram_clustering 8 10 12 \
#     --processes 4 --seeds 0
# 
#   done

STEP=0.375
SUBJECT_ID=sub-1006
EXPERIMENT=PFT_ISMRM2015Exp1
BASE=${TRACK_TO_LEARN_DATA}/datasets/tractoinferno/$SUBJECT_ID
npv=10

ID=2023-02-24-17_45_03

for seed in "${seeds[@]}";
do

  OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk

  BASE=${DATASET_FOLDER}/datasets/tractoinferno/${SUBJECT_ID}
  SCORING_DATA=${BASE}/scoring_data
  DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
  mkdir -p $DEST_FOLDER

  scil_compute_pft.py \
    $BASE/fodf/${SUBJECT_ID}__fodf.nii.gz \
    $BASE/mask/${SUBJECT_ID}__mask_wm.nii.gz \
    $BASE/maps/${SUBJECT_ID}__map_include.nii.gz \
    $BASE/maps/${SUBJECT_ID}__map_exclude.nii.gz  \
    $DEST_FOLDER/$OUT \
    --npv $npv --min_length $MIN \
    --max_length $MAX --step $STEP \
    --seed $seed -f -v

  validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}

  mkdir -p $validation_folder

  mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/

  # scil_recognize_multi_bundles.py \
  #   $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
  #   $SCORING_DATA/config/config_ind.json \
  #   $SCORING_DATA/atlas/*/ \
  #   $SCORING_DATA/output0GenericAffine.mat \
  #   --out $validation_folder/voting_results \
  #   -f --log_level DEBUG --multi_parameters 27 \
  #   --minimal_vote 0.4 --tractogram_clustering 8 10 12 \
  #   --processes 4 --seeds 0

done

SUBJECT_ID=sub-1006
EXPERIMENT=PFT_ISMRM2015Exp2
npv=20

ID=2023-02-24-21_08_25

for seed in "${seeds[@]}";
do

  OUT=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk

  BASE=${DATASET_FOLDER}/datasets/${SUBJECT_ID}
  SCORING_DATA=${BASE}/scoring_data
  DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$seed"
  mkdir -p $DEST_FOLDER

  scil_compute_pft.py \
    $BASE/fodf/${SUBJECT_ID}__fodf.nii.gz \
    $BASE/maps/${SUBJECT_ID}__interface.nii.gz \
    $BASE/maps/${SUBJECT_ID}__map_include.nii.gz \
    $BASE/maps/${SUBJECT_ID}__map_exclude.nii.gz  \
    $DEST_FOLDER/$OUT \
    --npv $npv --min_length $MIN \
    --max_length $MAX --step $STEP \
    --seed $seed -f -v

  validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}

  mkdir -p $validation_folder

  mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/

  # scil_recognize_multi_bundles.py \
  #   $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
  #   $SCORING_DATA/config/config_ind.json \
  #   $SCORING_DATA/atlas/*/ \
  #   $SCORING_DATA/output0GenericAffine.mat \
  #   --out $validation_folder/voting_results \
  #   -f --log_level DEBUG --multi_parameters 27 \
  #   --minimal_vote 0.4 --tractogram_clustering 8 10 12 \
  #   --processes 4 --seeds 0

done

