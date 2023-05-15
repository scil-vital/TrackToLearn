#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch

from TrackToLearn.trainers.vpg_train import (
    parse_args,
    VPGTrackToLearnTraining)

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
            "entropy_loss_coeff": {
                "type": "discrete",
                "values": [0.001]},
            },


        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "VC",
            "objective": "maximize",
        },
    }

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config)

    for experiment in opt.get_experiments(project_name=args.experiment):
        experiment.auto_metric_logging = False
        experiment.workspace = 'TrackToLearn'
        experiment.parse_args = False
        experiment.disabled = not args.use_comet

        lr = experiment.get_parameter("lr")
        gamma = experiment.get_parameter("gamma")
        entropy_loss_coeff = experiment.get_parameter("entropy_loss_coeff")

        arguments = vars(args)
        arguments.update({
            'lr': lr,
            'gamma': gamma,
            'entropy_loss_coeff': entropy_loss_coeff,
        })
        vpg_experiment = VPGTrackToLearnTraining(
            arguments,
            experiment,
        )
        vpg_experiment.run()


if __name__ == '__main__':
    main()
