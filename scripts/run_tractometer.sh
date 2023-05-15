# $1 Input tractogram
# $2 Scoring data
# $3 Output folder

python scripts/score_tractogram.py $1 \
  $2 \
  $3 \
  --save_full_vc \
  --save_full_ic \
  --save_full_nc \
  --save_ib \
  --save_vb -f -v
