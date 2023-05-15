#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import traceback
import torch

from TrackToLearn.trainers.acktr_train import (
    parse_args,
    ACKTRTrackToLearnTraining)

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
                "values": [0.01, 0.1, 0.15, 0.2, 0.25]},
            "gamma": {
                "type": "discrete",
                "values": [0.5, 0.75, 0.85, 0.90, 0.95, 0.99]},
            "delta": {
                "type": "discrete",
                "values": [0.0001, 0.0005, 0.001, 0.005, 0.01]},
            "entropy_loss_coeff": {
                "type": "discrete",
                "values": [0.001]},
            "lmbda": {
                "type": "discrete",
                "values": [0.95]},
        },
        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "VC",
            "objective": "maximize",
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
        delta = experiment.get_parameter("delta")
        lmbda = experiment.get_parameter("lmbda")
        entropy_loss_coeff = experiment.get_parameter("entropy_loss_coeff")

        arguments = vars(args)
        arguments.update({
            'lr': lr,
            'gamma': gamma,
            'lmbda': lmbda,
            'entropy_loss_coeff': entropy_loss_coeff,
            'delta': delta
        })
        # ACKTR is unstable learning
        try:
            acktr_experiment = ACKTRTrackToLearnTraining(
                arguments,
                experiment,
            )
            acktr_experiment.run()
        except RuntimeError as e:  # noqa: F841
            traceback.print_exc()
        except ValueError as v:  # noqa: F841
            traceback.print_exc()


if __name__ == '__main__':
    main()
