for s in /home/local/USHERBROOKE/thea1603/data/hcp105/derivatives/sf-tractomics/sub-*/; do
  sub=$(basename "$s" /)
  jq -cn \
     --arg key "$sub" \
     --arg inputs "$(realpath "$s/dwi/${sub}_model-tensor_param-tensor_dwimap.nii.gz")" \
     --arg peaks "$(realpath $s/dwi/${sub}_model-csd_param-peaks_dwimap.nii.gz)" \
     --arg tracking "$(realpath $s/dwi/${sub}_space-DWI_label-tracking_desc-local_mask.nii.gz)" \
     --arg seeding "/home/local/USHERBROOKE/thea1603/data/hcp105/derivatives/pft_maps/${sub}/${sub}__interface.nii.gz" \
     --arg anat "$(realpath $s/anat/${sub}_space-dwi_desc-preproc_T1w.nii.gz)" \
     '{($key): {inputs: [$inputs], peaks: $peaks, tracking: $tracking, seeding: $seeding, anat: $anat}}'
done | jq -s add > output.json
