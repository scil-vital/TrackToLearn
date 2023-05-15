#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch

from TrackToLearn.trainers.sac_train import (
    parse_args,
    SACTrackToLearnTraining)

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
            "lr": {
                "type": "discrete",
                "values": [5e-5, 1e-5, 5e-4, 1e-4, 1e-3, 5e-3]},
            "gamma": {
                "type": "discrete",
                "values": [0.5, 0.75, 0.85, 0.90, 0.95, 0.99]},
            "alpha": {
                "type": "discrete",
                "values": [0.075, 0.1, 0.15, 0.2, 0.3]},
        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "Reward",
            "objective": "maximize",
            "seed": args.rng_seed,
        },
    }

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config, project_name=args.experiment)

    for experiment in opt.get_experiments():
        experiment.auto_metric_logging = False
        experiment.workspace = 'TrackToLearn'
        experiment.parse_args = False
        experiment.disabled = not args.use_comet

        lr = experiment.get_parameter("lr")
        gamma = experiment.get_parameter("gamma")
        alpha = experiment.get_parameter("alpha")

        arguments = vars(args)
        arguments.update({
            'lr': lr,
            'gamma': gamma,
            'alpha': alpha
        })

        sac_experiment = SACTrackToLearnTraining(
            arguments,
            experiment
        )
        sac_experiment.run()


if __name__ == '__main__':
    main()
