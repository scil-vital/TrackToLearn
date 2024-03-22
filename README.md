# Track-to-Learn/TractOracle-RL: reinforcement learning for tractography.

See [Versions](#versions) for past and current interations.

## Getting started

### Installation and setup

**Right now, only python 3.10 is supported.**

It is recommended to use `virtualenv` to run the code

``` bash
virtualenv .env --python=python3.10
source .env/bin/activate
```

Then, install the dependencies and setup the repo with

``` bash
./install.sh
```

Getting errors during installation ? Open an issue !

### Tracking

You will need a trained agent for tracking. One is provided in the `models` folder. You can then track by running `ttl_track.py`.

```
usage: ttl_track.py [-h] [--input_wm] [--sh_basis {descoteaux07,tournier07}]
                    [--compress thresh] [-f] [--save_seeds] [--agent AGENT]
                    [--hyperparameters HYPERPARAMETERS] [--n_actor N]
                    [--npv NPV] [--min_length m] [--max_length M] [--prob %]
                    [--noise sigma] [--fa_map FA_MAP]
                    [--binary_stopping_threshold BINARY_STOPPING_THRESHOLD]
                    [--rng_seed RNG_SEED]
                    in_odf in_seed in_mask out_tractogram

 Generate a tractogram from a trained model. See `--help` for usage.
```

You will need to provide fODFs, a seeding mask and a WM mask. The seeding mask **must** represent the interface of white matter and gray matter. _WM tracking is no longer supported._

Agents used for tracking are constrained by their training regime. For example, the agents provided in `example_models` were trained on a volume with a resolution of ~1mm iso voxels and a step size of 0.75mm using fODFs of order 8, `descoteaux07` basis. When tracking on arbitrary data, the step-size and fODF order and basis will be adjusted accordingly automatically (i.e resulting in a step size of 0.375mm on 0.5mm iso diffusion data). **However**, if using fODFs in the `tournier07` basis, you will need to set the `--sh_basis` argument accordingly.

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

Example datasets and config files are available here: COMING SOON 

Then, you may train an agent by running `python TrackToLearn/trainers/sac_auto_train.py`.

```
usage: sac_auto_train.py [-h] [--workspace WORKSPACE] [--rng_seed RNG_SEED]
                         [--use_comet] [--n_dirs N_DIRS]
                         [--binary_stopping_threshold BINARY_STOPPING_THRESHOLD]
                         [--n_actor N_ACTOR] [--hidden_dims HIDDEN_DIMS]
                         [--load_agent LOAD_AGENT] [--max_ep MAX_EP]
                         [--log_interval LOG_INTERVAL] [--lr LR] [--gamma GAMMA]
                         [--alignment_weighting ALIGNMENT_WEIGHTING] [--npv NPV]
                         [--theta THETA] [--min_length m] [--max_length M]
                         [--step_size STEP_SIZE] [--prob %] [--noise sigma]
                         [--oracle_checkpoint ORACLE_CHECKPOINT]
                         [--oracle_validator] [--oracle_stopping_criterion]
                         [--sparse_oracle_weighting SPARSE_ORACLE_WEIGHTING]
                         [--scoring_data SCORING_DATA]
                         [--tractometer_reference TRACTOMETER_REFERENCE]
                         [--tractometer_validator]
                         [--tractometer_dilate TRACTOMETER_DILATE]
                         [--alpha ALPHA] [--batch_size BATCH_SIZE]
                         [--replay_size REPLAY_SIZE]
                         path experiment id dataset_file
sac_auto_train.py: error: the following arguments are required: path, experiment, id, dataset_file
```

Other trainers are available in `TrackToLearn/trainers`.

You can recreate an experiment by running a script in the `scripts` folder. These scripts should provide an excellent starting point for improving upon this work. You will only need to first set the `TRACK_TO_LEARN_DATA` environment variable to where you extracted the datasets (i.e. a network disk or somewhere with lots of space) and the `LOCAL_TRACK_TO_LEARN_DATA` environment variable your working folder (i.e. a faster local disk). Then, the script can be launched.

To use [Comet.ml](https://www.comet.ml/), follow instructions [here](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration), with the config file either in your home folder or current folder. **Usage of comet-ml is necessary for hyperparameter search**. This constraint may be removed in future releases.

## Contributing

Contributions are welcome ! There are several TODOs sprinkled through the project which may inspire you. A lot of the code's architecure could be improved, better organized, split and reworked to make the code cleaner. Several performance improvements could also easily be added.

See `ARCHITECURE.md` for an overview of the code.

## Versions

### TractOracle-RL (2024b)

> Théberge, A., Descoteaux, M., & Jodoin, P. M. (2024). TractOracle: towards an anatomically-informed reward function for RL-based tractography. Submitted to MICCAI 2024.

The reference commit to the `main` brain for this work is `TODO`. Please use this commit as starting point if you want to build upon Track-to-Learn (TractOracle-RL). See README above for usage.

Preprint comming soon.

Submitted work (hopefully) coming soon.

### What matters in Reinforcement Learning for Tractography (2024a)

> Théberge, A., Desrosiers, C., Boré, A., Descoteaux, M., & Jodoin, P. M. (2024). What matters in reinforcement learning for tractography. Medical Image Analysis, 93, 103085.

The reference commit to the `main` branch for this work is `9f97eefbbdb05a2c90ea74e8384ac2891b194a3e`. Please use this commit as starting point if you want to reproduce or build upon the work of the 2024a paper.

See journal paper: https://www.sciencedirect.com/science/article/pii/S1361841524000100

See preprint: https://arxiv.org/abs/2305.09041

### Incorporating anatomical priors into Track-to-Learn (2022)

The reference commit to the main branch for this work is `dbae9305b4a3e9f21c3249121ef5dc5ed9faa899`.

This work is presented at the *ISMRM Workshop on Diffusion MRI: From Research to Clinic*, poster \#34 (email me for the abstract and poster). This work adds the use of Continuous Map Criterion (CMC, https://www.sciencedirect.com/science/article/pii/S1053811914003541), asymmetric fODFs (https://archive.ismrm.org/2021/0865.html). They can be used with the `--cmc` and `--asymmetric` options respectively. Data and trained models are available here: https://zenodo.org/record/7153362

Dataset files are in `raw` and weights and results are in `experiments`. Results can be replicated using bash scripts (`sac_auto_train[_cmc|_asym|_cmc_asym].sh`) in the `scripts` folder of the code. The `DATASET_FOLDER` variable must be initialized to the folder where the `raw` and `experiments` folders are.

### Track-to-Learn (2021)

> Théberge, A., Desrosiers, C., Descoteaux, M., & Jodoin, P. M. (2021). Track-to-learn: A general framework for tractography with deep reinforcement learning. Medical Image Analysis, 72, 102093.
>
>
The reference commit to the main branch for this work is `e5f2e6008e499f46af767940b5b1eec7f9293859`.

See published version: https://www.sciencedirect.com/science/article/pii/S1361841521001390

See preprint: https://www.biorxiv.org/content/10.1101/2020.11.16.385229v1

A bug in the original implementation prevents the reproduction of the published results. The results presented in "What matters in Reinforcement Learning for Tractography" should be reproducible and considered as reference.

## How to cite

If you want to reference this work, please use (at least) one of

```
@article{theberge2024matters,
  title={What matters in reinforcement learning for tractography},
  author={Th{\'e}berge, Antoine and Desrosiers, Christian and Bor{\'e}, Arnaud and Descoteaux, Maxime and Jodoin, Pierre-Marc},
  journal={Medical Image Analysis},
  volume={93},
  pages={103085},
  year={2024},
  publisher={Elsevier}
}
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
}
```
