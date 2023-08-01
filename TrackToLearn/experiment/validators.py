from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs

from os.path import join as pjoin


class Validator(object):

    def __init__(self):

        self.name = ''

    def __call__(self, filename):

        assert False, 'not implemented'


class TractometerValidator(Validator):

    def __init__(self, base_dir):

        self.name = 'Tractometer'
        self.base_dir = base_dir

    def __call__(self, filename):

        #  Load bundle attributes for tractometer
        # TODO: No need to load this every time, should only be loaded
        # once
        gt_bundles_attribs_path = pjoin(
            self.base_dir, 'gt_bundles_attributes.json')
        basic_bundles_attribs = load_attribs(gt_bundles_attribs_path)

        # Score tractogram
        scores = score_submission(
            filename,
            self.base_dir,
            basic_bundles_attribs,
            compute_ic_ib=True)
        cleaned_scores = {}
        for k, v in scores.items():
            if type(v) in [float, int]:
                cleaned_scores.update({k: v})
        return cleaned_scores
