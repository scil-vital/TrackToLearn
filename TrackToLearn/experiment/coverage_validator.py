import numpy as np
from dipy.io.streamline import load_tractogram
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from TrackToLearn.experiment.validators import Validator


class CoverageValidator(Validator):

    def __init__(self):

        self.name = 'Coverage'

    def __call__(self, filename, env):

        wm_mask = env.tracking_mask.data
        count = np.count_nonzero(wm_mask)

        # Bbox check=False, TractoInferno volume may be cropped really tight
        sft = load_tractogram(filename, env.reference,
                              bbox_valid_check=False, trk_header_check=True)
        _, dimensions, _, _ = sft.space_attributes
        sft.to_vox()
        sft.to_corner()

        streamline_count = compute_tract_counts_map(sft.streamlines,
                                                    dimensions)
        streamline_count[streamline_count > 0] = 1
        coverage = np.count_nonzero(streamline_count)
        return {'Coverage': float(coverage / count)}
