# Track-to-Learn: A general framework for tractography with deep reinforcement learning

See preprint: https://www.biorxiv.org/content/10.1101/2020.11.16.385229v1
See published version: https://www.sciencedirect.com/science/article/pii/S1361841521001390

## How to cite

If you want to reference this work, please use

```
@article{theberge2021,
title = {Track-to-Learn: A general framework for tractography with deep reinforcement learning},
journal = {Medical Image Analysis},
pages = {102093},
year = {2021},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102093},
url = {https://www.sciencedirect.com/science/article/pii/S1361841521001390},
author = {Antoine Théberge and Christian Desrosiers and Maxime Descoteaux and Pierre-Marc Jodoin},
keywords = {Tractography, Deep Learning, Reinforcement Learning},
abstract = {Diffusion MRI tractography is currently the only non-invasive tool able to assess the white-matter structural connectivity of a brain. Since its inception, it has been widely documented that tractography is prone to producing erroneous tracks while missing true positive connections. Recently, supervised learning algorithms have been proposed to learn the tracking procedure implicitly from data, without relying on anatomical priors. However, these methods rely on curated streamlines that are very hard to obtain. To remove the need for such data but still leverage the expressiveness of neural networks, we introduce Track-To-Learn: A general framework to pose tractography as a deep reinforcement learning problem. Deep reinforcement learning is a type of machine learning that does not depend on ground-truth data but rather on the concept of “reward”. We implement and train algorithms to maximize returns from a reward function based on the alignment of streamlines with principal directions extracted from diffusion data. We show competitive results on known data and little loss of performance when generalizing to new, unseen data, compared to prior machine learning-based tractography algorithms. To the best of our knowledge, this is the first successful use of deep reinforcement learning for tractography.}
}
```

## Installation and setup

It is recommended to use a virtualenv to run the code

``` bash
virtualenv .env
```

Then, install the dependencies and setup the repo with

``` bash
pip install -r requirements.txt
pip install -e .
```

You will need to install extra dependencies to score the tractograms

``` bash
pip install -r extra-requirements.txt
```

Right now, only python 3.7 is supported.

## Running

First, make a dataset `.hdf5` file with `TrackToLearn/dataset/create_dataset.py`.

Then, you may train the TD3 agent by running `python TrackToLearn/runners/td3_train.py`.
```
usage: td3_train.py [-h] [--use_gpu] [--rng_seed RNG_SEED] [--use_comet]
                    [--run_tractometer] [--render]
                    [--ground_truth_folder GROUND_TRUTH_FOLDER]
                    [--n_signal N_SIGNAL] [--n_dirs N_DIRS]
                    [--add_neighborhood ADD_NEIGHBORHOOD]
                    [--n_seeds_per_voxel N_SEEDS_PER_VOXEL]
                    [--max_angle MAX_ANGLE] [--min_length MIN_LENGTH]
                    [--max_length MAX_LENGTH]
                    [--alignment_weighting ALIGNMENT_WEIGHTING]
                    [--straightness_weighting STRAIGHTNESS_WEIGHTING]
                    [--length_weighting LENGTH_WEIGHTING]
                    [--target_bonus_factor TARGET_BONUS_FACTOR]
                    [--exclude_penalty_factor EXCLUDE_PENALTY_FACTOR]
                    [--angle_penalty_factor ANGLE_PENALTY_FACTOR]
                    [--step_size STEP_SIZE] [--cmc] [--asymmetric]
                    [--recurrent RECURRENT] [--hidden_dims HIDDEN_DIMS]
                    [--load_policy LOAD_POLICY] [--max_ep MAX_EP]
                    [--log_interval LOG_INTERVAL] [--lr LR] [--gamma GAMMA]
                    [--tracking_batch_size TRACKING_BATCH_SIZE]
                    [--valid_noise VALID_NOISE] [--interface_seeding]
                    [--no_retrack] [--stochastic] [--action_std ACTION_STD]
                    [--training_batch_size TRAINING_BATCH_SIZE]
                    path experiment name dataset_file subject_id
                    test_dataset_file test_subject_id reference_file

```

Similarly, you may train the SAC or SAC with automatic entropy tuning using `sac_train.py` or `sac_auto_train.py`.

To use [Comet.ml](https://www.comet.ml/), follow instructions [here](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration), with the config file either in your home folder or current folder

The option for recurrent agents is there but recurrent agents are not yet implemented. Only single-subject training is supported. 

Once your agent is trained, you may track using `python TrackToLearn/runners/track.py` (directly on files) or `python TrackToLearn/runners/test.py` (from a dataset).

```bash
usage: track.py [-h] [--use_gpu] [--rng_seed RNG_SEED] [--use_comet]
                [--run_tractometer] [--render]
                [--ground_truth_folder GROUND_TRUTH_FOLDER]
                [--out_tractogram OUT_TRACTOGRAM]
                [--remove_invalid_streamlines] [--fa_map FA_MAP]
                [--n_signal N_SIGNAL] [--n_dirs N_DIRS]
                [--add_neighborhood ADD_NEIGHBORHOOD]
                [--n_seeds_per_voxel N_SEEDS_PER_VOXEL]
                [--max_angle MAX_ANGLE] [--min_length MIN_LENGTH]
                [--max_length MAX_LENGTH]
                [--alignment_weighting ALIGNMENT_WEIGHTING]
                [--straightness_weighting STRAIGHTNESS_WEIGHTING]
                [--length_weighting LENGTH_WEIGHTING]
                [--target_bonus_factor TARGET_BONUS_FACTOR]
                [--exclude_penalty_factor EXCLUDE_PENALTY_FACTOR]
                [--angle_penalty_factor ANGLE_PENALTY_FACTOR]
                [--step_size STEP_SIZE] [--cmc] [--asymmetric]
                [--tracking_batch_size TRACKING_BATCH_SIZE]
                [--valid_noise VALID_NOISE] [--interface_seeding]
                [--no_retrack] [--stochastic]
                path experiment name signal_file peaks_file seeding_file
                tracking_file target_file include_file exclude_file subject_id
                reference_file policy hyperparameters
```

```bash
usage: test.py [-h] [--use_gpu] [--rng_seed RNG_SEED] [--use_comet]
               [--run_tractometer] [--render]
               [--ground_truth_folder GROUND_TRUTH_FOLDER]
               [--remove_invalid_streamlines] [--fa_map FA_MAP]
               [--test_max_angle TEST_MAX_ANGLE] [--n_signal N_SIGNAL]
               [--n_dirs N_DIRS] [--add_neighborhood ADD_NEIGHBORHOOD]
               [--n_seeds_per_voxel N_SEEDS_PER_VOXEL] [--max_angle MAX_ANGLE]
               [--min_length MIN_LENGTH] [--max_length MAX_LENGTH]
               [--alignment_weighting ALIGNMENT_WEIGHTING]
               [--straightness_weighting STRAIGHTNESS_WEIGHTING]
               [--length_weighting LENGTH_WEIGHTING]
               [--target_bonus_factor TARGET_BONUS_FACTOR]
               [--exclude_penalty_factor EXCLUDE_PENALTY_FACTOR]
               [--angle_penalty_factor ANGLE_PENALTY_FACTOR]
               [--step_size STEP_SIZE] [--cmc] [--asymmetric]
               [--tracking_batch_size TRACKING_BATCH_SIZE]
               [--valid_noise VALID_NOISE] [--interface_seeding]
               [--no_retrack] [--stochastic]
               path experiment name dataset_file subject_id reference_file
               policy hyperparameters
```

## Track-to-Learn

A bug in the original implementation prevents the reproduction of the published results. **However** I am currently working on an updated and expanded version of the paper, which should be submitted in the comings months. For now, you should refer to the next section.

## Incorporating anatomical priors into Track-to-Learn

The reference commit to the master branch for this work is `dbae9305b4a3e9f21c3249121ef5dc5ed9faa899`.

This work is presented at the *ISMRM Workshop on Diffusion MRI: From Research to Clinic*, poster \#34. This work adds the use of Continuous Map Criterion (CMC, https://www.sciencedirect.com/science/article/pii/S1053811914003541), asymmetric fODFs (https://archive.ismrm.org/2021/0865.html) and interface, WM/GM boundary seeding. They can be used with the `--cmc`, `--asymmetric` and `--interface_seeding` options respectively. Data and trained models are available here: https://zenodo.org/record/7153362

Dataset files are in `raw` and weights and results are in `experiments`. Results can be replicated using bash scripts (`sac_auto_train[_cmc|_asym|_cmc_asym].sh`) in the `scripts` folder of the code. The `DATASET_FOLDER` variable must be initialized to the folder where the `raw` and `experiments` folders are.
