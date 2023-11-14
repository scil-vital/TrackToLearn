# Track-to-Learn: A general framework for tractography with deep reinforcement learning

## Getting started

### Installation and setup

**Right now, only python 3.8 is supported.**

It is recommended to use `virtualenv` to run the code

``` bash
virtualenv .env --python=python3.8
source .env/bin/activate
```

Then, install the dependencies and setup the repo with

``` bash
# Install common requirements

# edit requirements.txt as needed to change your torch install
pip install -r requirements.txt
# Install some specific requirements directly from git
# scilpy 1.3.0 requires a deprecated version of sklearn on pypi
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install git+https://github.com/scilus/scilpy@1.3.0#egg=scilpy
pip install git+https://github.com/scil-vital/dwi_ml@70b9a97f85d295b0f03388ddb3c63b3da120ada3
pip install git+https://github.com/scilus/ismrm_2015_tractography_challenge_scoring.git
# Load the project into the environment
pip install -e .
```

TrackToLearn was developed using `torch==1.9.1` with CUDA 11. You may have to change the torch version in `requirements.txt` to suit your local installation (i.e CPU-only `torch` or using CUDA 10).

Still getting errors during installation ? See the wiki: [https://github.com/scil-vital/TrackToLearn/wiki/Troubleshooting](https://github.com/scil-vital/TrackToLearn/wiki/Troubleshooting) or open an issue !

### Tracking

You will need a trained agent for tracking. One is provided in the `example_model` folder. You can then track by running `ttl_track.py`.

```
usage: ttl_track.py [-h] [--sh_basis {descoteaux07,tournier07}] [--compress thresh] [-f] [--save_seeds]
                    [--policy POLICY] [--hyperparameters HYPERPARAMETERS] [--npv NPV] [--interface]
                    [--min_length m] [--max_length M] [--prob sigma] [--fa_map FA_MAP] [--n_actor N]
                    [--rng_seed RNG_SEED]
                    in_odf in_seed in_mask out_tractogram
```

You will need to provide fODFs, a seeding mask and a WM mask.

Agents used for tracking are constrained by their training regime. For example, the agents provided in `example_models` were trained on a volume with a resolution of 2mm iso voxels and a step size of 0.75mm using fODFs of order 6, `descoteaux07` basis. When tracking on arbitrary data, the step-size and fODF order and basis will be adjusted accordingly automatically. **However**, if using fODFs in the `tournier07` (coming from MRtrix, for example), you will need to set the `--sh_basis` argument accordingly.

Other trained agents are available here: https://zenodo.org/record/7853590

### Training

First, make a dataset `.hdf5` file with `TrackToLearn/dataset/create_dataset.py`.
```
usage: create_dataset.py [-h] [--normalize] path config_file output

positional arguments:
  path         Location of the dataset files.
  config_file  Configuration file to load subjects and their volumes.
  output       Output filename including path

optional arguments:
  -h, --help   show this help message and exit
  --normalize  If set, normalize first input signal.
```

Example datasets and config files are available here: https://zenodo.org/record/7853832

Then, you may train a PPO agent, for example, by running `python TrackToLearn/trainers/ppo_train.py`.

```
usage: ppo_train.py [-h] [--use_gpu] [--rng_seed RNG_SEED] [--use_comet]
                    [--run_tractometer] [--render] [--n_signal N_SIGNAL]
                    [--n_dirs N_DIRS] [--add_neighborhood ADD_NEIGHBORHOOD]
                    [--cmc] [--asymmetric] [--n_actor N_ACTOR]
                    [--hidden_dims HIDDEN_DIMS] [--load_policy LOAD_POLICY]
                    [--max_ep MAX_EP] [--log_interval LOG_INTERVAL] [--lr LR]
                    [--gamma GAMMA]
                    [--alignment_weighting ALIGNMENT_WEIGHTING]
                    [--straightness_weighting STRAIGHTNESS_WEIGHTING]
                    [--length_weighting LENGTH_WEIGHTING]
                    [--target_bonus_factor TARGET_BONUS_FACTOR]
                    [--exclude_penalty_factor EXCLUDE_PENALTY_FACTOR]
                    [--angle_penalty_factor ANGLE_PENALTY_FACTOR]
                    [--npv N_SEEDS_PER_VOXEL]
                    [--theta MAX_ANGLE] [--min_length MIN_LENGTH]
                    [--max_length MAX_LENGTH] [--step_size STEP_SIZE]
                    [--prob VALID_NOISE] [--interface_seeding]
                    [--no_retrack] [--entropy_loss_coeff ENTROPY_LOSS_COEFF]
                    [--action_std ACTION_STD] [--lmbda LMBDA]
                    [--K_epochs K_EPOCHS] [--eps_clip EPS_CLIP]
                    path experiment id dataset_file subject_id
                    test_dataset_file test_subject_id reference_file
                    scoring_data
```

Other trainers are available in `TrackToLearn/trainers`.

You can recreate an experiment by running a script in the `scripts` folder. These scripts should provide an excellent starting point for improving upon this work. You will only need to first set the `TRACK_TO_LEARN_DATA` environment variable to where you extracted the datasets (i.e. a network disk or somewhere with lots of space) and the `LOCAL_TRACK_TO_LEARN_DATA` environment variable your working folder (i.e. a faster local disk). Then, the script can be launched.

To use [Comet.ml](https://www.comet.ml/), follow instructions [here](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration), with the config file either in your home folder or current folder. **Usage of comet-ml is necessary for hyperparameter search**, but this constraint should be removed in future releases.

The option for recurrent agents is there but recurrent agents are not yet implemented. Training and validation can be performed on different subjects, but training (or validation) on multiple subjects is not yet supported. 

## Contributing

Contributions are welcome ! There are several TODOs sprinkled through the project which may inspire you. A lot of the code's architecure could be improved, better organized, split and reworked to make the code cleaner. Several performance improvements could also easily be added.

See `ARCHITECURE.md` for an overview of the code.

## What matters in Reinforcement Learning for Tractography (2023)

The reference commit to the `main` branch for this work is `9f97eefbbdb05a2c90ea74e8384ac2891b194a3e`.

See preprint: https://arxiv.org/abs/2305.09041

This version of Track-to-Learn should serve as a reference going forward to use and improve upon Track-to-Learn. Refer to the readme above for usage.

## Incorporating anatomical priors into Track-to-Learn (2022)

The reference commit to the master branch for this work is `dbae9305b4a3e9f21c3249121ef5dc5ed9faa899`.

This work is presented at the *ISMRM Workshop on Diffusion MRI: From Research to Clinic*, poster \#34 (email me for the abstract and poster). This work adds the use of Continuous Map Criterion (CMC, https://www.sciencedirect.com/science/article/pii/S1053811914003541), asymmetric fODFs (https://archive.ismrm.org/2021/0865.html). They can be used with the `--cmc` and `--asymmetric` options respectively. Data and trained models are available here: https://zenodo.org/record/7153362

Dataset files are in `raw` and weights and results are in `experiments`. Results can be replicated using bash scripts (`sac_auto_train[_cmc|_asym|_cmc_asym].sh`) in the `scripts` folder of the code. The `DATASET_FOLDER` variable must be initialized to the folder where the `raw` and `experiments` folders are.

## Track-to-Learn (2021)

See published version: https://www.sciencedirect.com/science/article/pii/S1361841521001390

See preprint: https://www.biorxiv.org/content/10.1101/2020.11.16.385229v1

A bug in the original implementation prevents the reproduction of the published results. The results presented in "What matters in Reinforcement Learning for Tractography" should be reproducible and considered as reference.

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
