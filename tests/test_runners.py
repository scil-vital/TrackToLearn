def test_ttl_track(script_runner):
    # Call 'ttl_track.py' from the command line and assert that it
    # runs without errors

    ret = script_runner.run('ttl_track.py', '--help')
    assert ret.success

def test_ttl_track_from_hdf5(script_runner):
    # Call 'ttl_track_from_hdf5.py' from the command line and assert that it
    # runs without errors

    ret = script_runner.run('ttl_track_from_hdf5.py', '--help')
    assert ret.success


def test_sac_auto_train(script_runner):
    # Call 'sac_auto_train.py' from the command line and assert that it
    # runs without errors

    ret = script_runner.run('TrackToLearn/trainers/sac_auto_train.py',
                            '--help')
    assert ret.success
