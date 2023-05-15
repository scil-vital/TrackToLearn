#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch
import traceback

from TrackToLearn.trainers.trpo_train import (
    parse_args,
    TRPOTrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)
    from comet_ml import Optimizer

    # We only need to specify the algorithm and hyperparameters to use:

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
            "lmbda": {
                "type": "discrete",
                "values": [0.95]},
            "delta": {
                "type": "discrete",
                "values": [0.001, 0.01, 0.1]},
            "K_epochs": {
                "type": "discrete",
                "values": [5]},
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
        lmbda = experiment.get_parameter("lmbda")
        entropy_loss_coeff = experiment.get_parameter("entropy_loss_coeff")

        K_epochs = experiment.get_parameter("K_epochs")
        delta = experiment.get_parameter("delta")

        arguments = vars(args)
        arguments.update({
            'lr': lr,
            'gamma': gamma,
            'lmbda': lmbda,
            'entropy_loss_coeff': entropy_loss_coeff,
            'K_epochs': K_epochs,
            'delta': delta
        })

        try:
            trpo_experiment = TRPOTrackToLearnTraining(
                arguments,
                experiment,
            )
            trpo_experiment.run()
        except RuntimeError as e:  # noqa: F841
            traceback.print_exc()
        except ValueError as v:  # noqa: F841
            traceback.print_exc()
        except comet_ml.exceptions.InterruptedExperiment:
            print('Experiment stopped by user')


if __name__ == '__main__':
    main()
