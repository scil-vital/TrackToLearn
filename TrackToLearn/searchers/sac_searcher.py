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
        "algorithm": "grid",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "lr": {
                "type": "discrete",
                "values": [5e-4, 1e-4, 5e-5, 1e-5]},
            "gamma": {
                "type": "discrete",
                "values": [0.75, 0.85, 0.90, 0.95, 0.99]},
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

        sac_experiment = SACTrackToLearnTraining(
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
            args.valid_noise,
            lr,
            gamma,
            alpha,
            # SAC
            args.training_batch_size,
            # Env params
            args.n_seeds_per_voxel,
            args.max_angle,
            args.min_length,
            args.max_length,
            args.step_size,  # Step size (in mm)
            args.alignment_weighting,
            args.straightness_weighting,
            args.length_weighting,
            args.target_bonus_factor,
            args.exclude_penalty_factor,
            args.angle_penalty_factor,
            args.tracking_batch_size,
            args.n_signal,
            args.n_dirs,
            args.interface_seeding,
            args.no_retrack,
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
        sac_experiment.run()


if __name__ == '__main__':
    main()
