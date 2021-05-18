# Track-to-Learn: A general framework for tractography with deep reinforcement learning

See preprint: https://www.biorxiv.org/content/10.1101/2020.11.16.385229v1

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
pip install -e .
```

You will need to install extra dependencies to score the tractograms

``` bash
pip install -r extra-requirements.txt
```

Right now, only python 3.7 is supported.

## Running

First, make a dataset `.hdf5` file with `TrackToLearn/dataset/create_dataset.py`.
Then, you may train the TD3 agent by running

```bash
python TrackToLearn/runners/td3_train.py \
  PATH \
  EXPERIMENT_NAME \
  EXPERIMENT_ID \
  DATASET_FILE \
  SUBJECT_ID \
  TEST_DATASET_FILE \
  TEST_SUBJECT_ID \
  REFERENCE_FILE
```

Similarly, you may train the SAC agent by running `TrackToLearn/dataset/create_dataset.py` using the same parameters. **Or run a script in the `scripts` folder.**

Then, you may track with your trained agent with

```bash
python TrackToLearn/runners/test.py \
  PATH \
  EXPERIMENT_NAME \
  EXPERIMENT_ID \
  TEST_DATASET_FILE \
  TEST_SUBJECT_ID \
  REFERENCE_FILE \
  PATH/model \
  PATH/model/hyperparameters.json
```

To use [Comet.ml](https://www.comet.ml/), follow instructions [here](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration), with the config file either in your home folder or current folder
