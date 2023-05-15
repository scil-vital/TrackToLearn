STEP=0.75
seeds=(1111 2222 3333 4444 5555)
BASE=${TRACK_TO_LEARN_DATA}/datasets/$DATASET

DATASET=fibercup
EXPERIMENT=PFT_FiberCup075
NPV=33
# 
for seed in "${seeds[@]}";
do

  OUT_FOLDER=${TRACK_TO_LEARN_DATA}/experiments/${EXPERIMENT}/${seed}/
  OUT=pft_${DATASET}_${NPV}_${STEP}_${seed}.trk

  scripts/score.sh $OUT_FOLDER/$OUT $OUT_FOLDER/scoring_0.0_${DATASET}_${NPV} ${DATASET}

done

EXPERIMENT=PFT_FiberCupGM075
NPV=33

for seed in "${seeds[@]}";
do

  OUT_FOLDER=${TRACK_TO_LEARN_DATA}/experiments/${EXPERIMENT}/${seed}/
  OUT=pft_${DATASET}_${NPV}_${STEP}_${seed}.trk

  scripts/score.sh $OUT_FOLDER/$OUT $OUT_FOLDER/scoring_0.0_${DATASET}_${NPV} ${DATASET}
done

DATASET=fibercup_flipped
EXPERIMENT=PFT_FiberCup075
BASE=${TRACK_TO_LEARN_DATA}/datasets/$DATASET
NPV=33

for seed in "${seeds[@]}";
do

  OUT_FOLDER=${TRACK_TO_LEARN_DATA}/experiments/${EXPERIMENT}/${seed}/
  OUT=pft_${DATASET}_${NPV}_${STEP}_${seed}.trk

  scripts/score.sh $OUT_FOLDER/$OUT $OUT_FOLDER/scoring_0.0_${DATASET}_${NPV} ${DATASET}
done

DATASET=fibercup_flipped
EXPERIMENT=PFT_FiberCupGM075
NPV=33

for seed in "${seeds[@]}";
do

  OUT_FOLDER=${TRACK_TO_LEARN_DATA}/experiments/${EXPERIMENT}/${seed}/

  OUT=pft_${DATASET}_${NPV}_${STEP}_${seed}.trk
  scripts/score.sh $OUT_FOLDER/$OUT $OUT_FOLDER/scoring_0.0_${DATASET}_${NPV} ${DATASET}

done

DATASET=ismrm2015
EXPERIMENT=PFT_ISMRM2015075
BASE=${TRACK_TO_LEARN_DATA}/datasets/$DATASET
NPV=7

for seed in "${seeds[@]}";
do

  OUT_FOLDER=${TRACK_TO_LEARN_DATA}/experiments/${EXPERIMENT}/${seed}/
  OUT=pft_${DATASET}_${NPV}_${STEP}_${seed}.trk

  scripts/score.sh $OUT_FOLDER/$OUT $OUT_FOLDER/scoring_0.0_${DATASET}_${NPV} ${DATASET}
done

DATASET=ismrm2015
EXPERIMENT=PFT_ISMRM2015GM075

for seed in "${seeds[@]}";
do

  OUT_FOLDER=${TRACK_TO_LEARN_DATA}/experiments/${EXPERIMENT}/${seed}/

  OUT=pft_${DATASET}_${NPV}_${STEP}_${seed}.trk
  scripts/score.sh $OUT_FOLDER/$OUT $OUT_FOLDER/scoring_0.0_${DATASET}_${NPV} ${DATASET}
done

