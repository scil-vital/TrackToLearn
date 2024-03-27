#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch

from TrackToLearn.trainers.sac_auto_train import (
    parse_args,
    SACAutoTrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)
    from comet_ml import Optimizer

    # We only need to specify the algorithm and hyperparameters to use:
    config = {
        # We pick the Bayes algorithm:
        "algorithm": "grid",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "gamma": {
                "type": "discrete",
                "values": [0.90, 0.95, 0.98, 0.99]},
            "oracle_bonus": {
                "type": "discrete",
                "values": [1.0, 5.0, 7.0, 10.0]}
        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "VC",
            "objective": "maximize",
            "seed": args.rng_seed,
            "retryLimit": 3,
            "retryAssignLimit": 3,
        },
    }

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config)

    for experiment in opt.get_experiments(project_name=args.experiment):
        experiment.auto_metric_logging = False
        experiment.workspace = args.workspace
        experiment.parse_args = False
        experiment.disabled = not args.use_comet

        oracle_bonus = experiment.get_parameter("oracle_bonus")
        gamma = experiment.get_parameter("gamma")

        arguments = vars(args)
        arguments.update({
            'oracle_bonus': oracle_bonus,
            'gamma': gamma,
        })

        sac_experiment = SACAutoTrackToLearnTraining(
            arguments,
            experiment
        )
        sac_experiment.run()


if __name__ == '__main__':
    main()
