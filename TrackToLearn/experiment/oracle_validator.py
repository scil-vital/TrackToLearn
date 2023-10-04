import numpy as np
from dipy.io.streamline import load_tractogram

from TrackToLearn.experiment.validators import Validator
from TrackToLearn.oracles.oracle import OracleSingleton


class OracleValidator(Validator):

    def __init__(self, checkpoint, reference, device):

        self.name = 'Oracle'
        self.reference = reference

        if checkpoint:
            self.checkpoint = checkpoint
            self.model = OracleSingleton(checkpoint, device)
        else:
            self.checkpoint = None

        self.device = device

    def __call__(self, filename):

        sft = load_tractogram(filename, self.reference,
                              bbox_valid_check=True, trk_header_check=True)

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
        return {'Oracle': float(np.mean(accuracy))}
