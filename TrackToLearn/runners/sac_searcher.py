#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch

from TrackToLearn.runners.sac_train import (
    parse_args,
    SACTrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)
    from comet_ml import Optimizer

    # We only need to specify the algorithm and hyperparameters to use:
    config = {
        # We pick the Bayes algorithm:
        "algorithm": "bayes",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "alpha": {"type": "float", "min": 0.01, "max": 0.3},
            "lr": {"type": "float", "min": 5e-5, "max": 5e-2},
            "gamma": {"type": "float", "min": 0.75, "max": 0.99},
        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "VC",
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

        gamma = experiment.get_parameter("gamma")
        alpha = experiment.get_parameter("alpha")
        lr = experiment.get_parameter("lr")

        td3_experiment = SACTrackToLearnTraining(
            # Dataset params
            args.path,
            args.experiment,
            args.name,
            args.dataset_file,
            args.subject_id,
            args.test_dataset_file,
            args.test_subject_id,
            args.reference_file,
            args.ground_truth_folder,
            # RL params
            args.max_ep,
            args.log_interval,
            args.action_std,
            args.valid_noise,
            lr,
            gamma,
            alpha,
            # TD3 params
            args.training_batch_size,
            # Env params
            args.n_seeds_per_voxel,
            args.max_angle,
            args.min_length,
            args.max_length,
            args.step_size,  # Step size (in mm)
            args.tracking_batch_size,
            args.n_signal,
            args.n_dirs,
            # Model params
            args.n_latent_var,
            args.hidden_layers,
            args.add_neighborhood,
            # Experiment params
            args.use_gpu,
            args.rng_seed,
            experiment,
            args.render,
            args.run_tractometer,
            args.load_teacher,
            args.load_policy,
        )
        td3_experiment.run()


if __name__ == '__main__':
    main()
