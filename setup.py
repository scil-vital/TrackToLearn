import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as f:
    required_dependencies = f.read().splitlines()
    external_dependencies = []
    torch_added = False
    for dependency in required_dependencies:
        external_dependencies.append(dependency)

setup(
    name='Track-to-Learn',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1',

    description='Deep reinforcement learning for tractography',
    long_description="",

    # The project's main homepage.
    url='https://github.com/scil-vital/TrackToLearn',

    # Author details
    author='Antoine Théberge',
    author_email='antoine.theberge@usherbrooke.ca',

    # Choose your license
    license='GNU General Public License v3.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='Deep Reinforcement Learning Tractography',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['TrackToLearn'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=external_dependencies,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[
        'models/last_model_state_critic.pth',
        'models/last_model_state_actor.pth',
        'models/hyperparameters.json',
    ],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            "ttl_track.py=TrackToLearn.runners.ttl_track:main",
            "ttl_track_from_hdf5.py=TrackToLearn.runners.ttl_track_from_hdf5:main"] # noqa E501
    },
    include_package_data=True,
)
