# Track-to-Learn

The overall structure of the project is

```
- TrackToLearn
  - algorithms
  - datasets
  - environments
  - experiment
  - runners
  - searchers
  - trainers
  - utils
- example_model
- scripts
- cc_scripts
```

In `TrackToLearn`, you will find the codebase for the project. In `scripts`, you will find all the scripts used to train agents in `What matters in ...`[^1] and `Incorporating anatomical priors into...`[^2]. In `cc_scripts`, you will find slurm scripts that have been used to do the architecture search of [^1]. `example_model` contains the weights and hyperparameters of an agent trained on the ISMRM2015 dataset.


The entry points for launching `TrackToLearn` are in `runners`, `trainers` or `searchers`. 

```
- TrackToLearn
  - runners
    - ttl_track.py
    - ttl_validation.py
  - searchers
    - a2c_searcher.py
    - acktr_searcher.py
    - ddpg_searcher.py
    - ppo_searcher.py
    - sac_auto_searcher.py
    - sac_searcher.py
    - td3_searcher.py
    - trpo_searcher.py
    - vpg_searcher.py
  - trainers
    - a2c_train.py
    - acktr_train.py
    - ddpg_train.py
    - ppo_train.py
    - sac_auto_train.py
    - sac_train.py
    - td3_train.py
    - trpo_train.py
    - vpg_train.py
```

The `runenrs` folder contains scripts for tracking either on a "dataset" (`ttl_validation.py`) or on arbitrary files (`ttl_track.py`, similarly to launching tracking in `scilpy`[^3]). These are also added to your PATH during installation. The `searchers` module contains scripts for launching an hyperparameter search for the relevant algorithm. The `trainers` module contains scripts for launching training for the relevant algorithm.

The `algorithms` module contains several implementations of RL algorithms.

```
- TrackToLearn
  - algorithms
    - rl.py
    - utils.py
    - a2c.py
    - acktr.py
    - ddpg.py
    - ppo.py
    - sac_auto.py
    - sac.py
    - td3.py
    - trpo.py
    - vpg.py
    - shared
      - onpolicy.py
      - offpolicy.py
      - replay.py
```

The `rl` submodule contains the core of all RL algorithms implementations and most things that are relevant to all (such as the RL loop at inference, for example). The `algorithms/utils` submodule contains functions relevant to most RL algorithms. The shared submodule mostly contains classes relevant to polices and critics. Other files are implementations of RL algorithms.

The `experiment` submodule contains core classes and functions for launching, monitoring and reproducing experiments. 

```
- TrackToLearn
  - experiment 
    - experiment.py
    - train.py
    - ttl.py
```

`experiment.py` contains the base class for experiments as well as most of the arguments used in entry-point scripts. `ttl.py` contains the base class for TrackToLearn experiments, which may be training or tracking or other. `train.py` contains the base class for training runs, from which "trainers" inherit.


The `enviroments` submodule contains everything related to RL environments.

```
- TrackToLearn
  - environments
    - env.py
    - interface_tracking_env.py
    - noisy_tracker.py
    - reward.py
    - tracker.py
    - utils.py
```

`env.py` contains the base abstract class for environments, `BaseEnv`, in Track-to-Learn. `tracker.py` contains several concrete classes that inherit from `BaseEnv`. `interface_tracking_env.py` and `noisy_tracker.py` contain classes that inherit from classes in `tracker`. `reward.py` contains the class handling the reward function.

Finally, the `datasets` submodule contains everything related to the creation and processing of datasets.

```
- TrackToLearn
  - datasets
    - create_dataset.py
    - processing.py
    - utils.py
```

The `create_dataset.py` script can be called to create a HDF5 containing training, validation and test subjects. `processing.py` contains util functions related to dataset creation.

[^1]: "What matters in reinforcement learning for tractography"
[^2]: Incorporating anatomical priors into Track-to-Learn, ISMRM Workshop on Diffusion MRI: From Research to Clinic, poster #34.
[^3]: scilpy: [https://github.com/scilus/scilpy](https://github.com/scilus/scilpy)
