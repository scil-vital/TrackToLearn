import numpy as np
from dipy.io.streamline import load_tractogram
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from TrackToLearn.experiment.validators import Validator
from TrackToLearn.oracles.oracle import OracleSingleton


class OracleValidator(Validator):

    def __init__(self, checkpoint, device):

        self.name = 'Oracle'

        if checkpoint:
            self.checkpoint = checkpoint
            self.model = OracleSingleton(checkpoint, device)
        else:
            self.checkpoint = None

        self.device = device

    def __call__(self, filename, env):

        # Bbox check=False, TractoInferno volume may be cropped really tight
        sft = load_tractogram(filename, env.reference,
                              bbox_valid_check=False, trk_header_check=True)
        _, dimensions, _, _ = sft.space_attributes
        wm_mask = env.tracking_mask.data
        count = np.count_nonzero(wm_mask)

        sft.to_vox()
        sft.to_corner()

        streamlines = sft.streamlines

        if len(streamlines) == 0:
            return {}

        batch_size = 4096
        N = len(streamlines)
        predictions = np.zeros((N))
        for i in range(0, N, batch_size):

            j = i + batch_size
            scores = self.model.predict(streamlines[i:j])
            predictions[i:j] = scores
        accuracy = (predictions > 0.5).astype(float)

        streamline_count = compute_tract_counts_map(
            sft.streamlines[predictions > 0.5], dimensions)

        streamline_count[streamline_count > 0] = 1
        coverage = np.count_nonzero(streamline_count)
        return {'Oracle': float(np.mean(accuracy)),
                'Coverage':  float(coverage / count)}
