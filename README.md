## Track-to-Learn: A general framework for tractography with deep reinforcement learning

See preprint: https://www.biorxiv.org/content/10.1101/2020.11.16.385229v1

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

Similarly, you may train the SAC agent by running `TrackToLearn/dataset/create_dataset.py` using the same parameters. *Or run a script in the `scripts` folder.*

Then, you may track with your trained agent with

```bash
python TrackToLearn/runners/test.py \
  PATH \
  EXPERIMENT_NAME \
  EXPERIMENT_ID \
  TEST_DATASET_FILE \
  TEST_SUBJECT_ID \
  REFERENCE_FILE \
  PATH/model" \
  PATH/model/hyperparameters.json"
```

To use [Comet.ml](https://www.comet.ml/), follow instructions [here](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration), with the config file either in your home folder or current folder

## Contributing

Please follow PEP8 conventions, use meaningful commit messages and mark PRs appropriately. Following `numpy` standards is heavily recommended: https://numpy.org/doc/1.16/dev/gitwash/development_workflow.html
