#!/bin/sh

if [ $# -eq 0 ] || [ $1 = '-h' ] || [ $1 = '--help' ]
then
    echo "-----"
    echo "Usage: "
    echo ">> scil_score_ismrm_Renauld2023.sh tractogram out_dir scoring_data"
    echo "-----"
    exit
fi

tractogram=$1
out_dir=$2
scoring_data=$3

config_file_segmentation=$scoring_data/config_file_segmentation.json
config_file_tractometry=$scoring_data/config_file_tractometry.json
ref=$scoring_data/ROI/all_masks/CA.nii.gz

if [ ! -f $tractogram ]
then
    echo "Tractogram $tractogram does not exist"
    exit
fi

if [ -d $out_dir ]
then
    echo "Out dir $out_dir already exists. Delete first."
    exit 1
fi


echo '------------- SEGMENTATION ------------'
scil_score_tractogram.py $tractogram $config_file_segmentation $out_dir --no_empty \
    --gt_dir $scoring_data --reference $ref --json_prefix tmp_ --no_bbox_check --unique --compute_ic -v;

echo '------------- Merging CC sub-bundles ------------'
CC_files=$(ls $out_dir/segmented_VB/CC* 2> /dev/null)
if [ "$CC_files" != '' ]
then
    scil_tractogram_math.py lazy_concatenate $CC_files $out_dir/segmented_VB/CC_VS.trk;
fi
echo '------------- Merging ICP left sub-bundles ------------'
ICP_left_files=$(ls $out_dir/segmented_VB/ICP_left* 2> /dev/null)
if [ "$ICP_left_files" != '' ]
then
    scil_tractogram_math.py lazy_concatenate $ICP_left_files $out_dir/segmented_VB/ICP_left_VS.trk;
fi
echo '------------- Merging ICP right sub-bundles ------------'
ICP_right_files=$(ls $out_dir/segmented_VB/ICP_right* 2> /dev/null)
if [ "$ICP_right_files" != '' ]
then
    scil_tractogram_math.py lazy_concatenate $ICP_right_files  $out_dir/segmented_VB/ICP_right_VS.trk;
fi

echo '------------- FINAL SCORING ------------'
scil_score_bundles.py -v $config_file_tractometry $out_dir \
      --gt_dir $scoring_data --reference $ref --no_bbox_check 

cat $out_dir/results.json
