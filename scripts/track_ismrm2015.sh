#!/usr/bin/bash

./scripts/rl_test.sh SAC_Auto_ISMRM2015TrainOracle oracle_2023-11-13-15_09_01
./scripts/rl_test_oracle_filter.sh SAC_Auto_ISMRM2015TrainOracle oracle_2023-11-13-15_09_01

./scripts/rl_test.sh SAC_Auto_ISMRM2015TrainOracle oracle_no_stop_2023-11-13-15_06_10
./scripts/rl_test_oracle_filter.sh SAC_Auto_ISMRM2015TrainOracle tractometer_2023-11-13-17_43_50

./scripts/rl_test.sh SAC_Auto_ISMRM2015TrainOracle tractometer_2023-11-13-17_43_50
./scripts/rl_test_oracle_filter.sh SAC_Auto_ISMRM2015TrainOracle tractometer_2023-11-13-17_43_50

./scripts/rl_test.sh SAC_Auto_ISMRM2015TrainOracle vanilla_2023-11-13-14_19_42
./scripts/rl_test_oracle_filter.sh SAC_Auto_ISMRM2015TrainOracle vanilla_2023-11-13-14_19_42
