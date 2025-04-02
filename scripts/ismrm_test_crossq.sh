python TrackToLearn/trainers/crossq_train.py \
    test_ismrm2015 \
    crossq \
    stop_oracle \
    ismrm2015/ismrm2015.hdf5 \
    --max_ep=10000 \
    --log_interval=50 \
    --rng_seed=5555 \
    --npv=2 \
    --theta=30 \
    --lr=0.0005 \
    --gamma=0.95 \
    --alignment_weighting=1.0 \
    --hidden_dims=1024-1024-1024 \
    --n_dirs=100 \
    --n_actor=4096 \
    --use_comet \
    --binary_stopping_threshold=0.1 \
    --tractometer_validator \
    --scoring_data=ismrm2015/scoring_data \
    --oracle_validator \
    --oracle_stopping_criterion \
    --oracle_checkpoint=models/epoch_49_ismrm2015v4.ckpt
