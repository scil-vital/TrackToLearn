import numpy as np
import torch

from nibabel.streamlines import Tractogram
from tqdm import tqdm
from typing import Tuple

from TrackToLearn.environments.env import BaseEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLAlgorithm(object):
    """
    Abstract sample-gathering and training algorithm.
    """

    def __init__(
        self,
        input_size: int,
        action_size: int = 3,
        hidden_size: int = 256,
        action_std: float = 0.35,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 10000,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda:0",
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_size: int
            Width of the NN
        action_std: float
            Starting standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        batch_size: int
            Batch size for replay buffer sampling
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        self.max_action = 1.
        self.t = 1

        self.action_std = action_std
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size

        self.rng = rng

    def _validation_episode(
        self,
        initial_state,
        env,
    ):
        """
        Main loop for the algorithm
        From a starting state, run the model until the env. says its done

        Parameters
        ----------
        initial_state: np.ndarray
            Initial state of the environment
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        tractogram: Tractogram
            Tractogram containing the tracked streamline
        running_reward: float
            Cummulative training steps reward
        """

        running_reward = 0
        state = initial_state
        tractogram = None
        done = False
        while not np.all(done):
            # Select action according to policy + noise to make tracking
            # probabilistic
            action = self.policy.select_action(
                np.array(state))

            # Perform action
            next_state, reward, done, _ = env.step(action)

            # Keep track of reward
            running_reward += sum(reward)

            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            # This line also set the next_state as the state
            new_tractogram, state, _ = env.harvest(next_state, compress=True)

            # Add streamlines to the lot
            if len(new_tractogram.streamlines) > 0:
                if tractogram is None:
                    tractogram = new_tractogram
                else:
                    tractogram += new_tractogram

        return tractogram, running_reward

    def generate_streamlines(
        self,
        batch_size,
        env: BaseEnv,
        backward_env: BaseEnv
    ):

        # Track for every seed in the environment
        for i, start in enumerate(tqdm(range(0, len(env.seeds), batch_size))):

            # Last batch might not be "full"
            end = min(start + batch_size, len(env.seeds))

            state = env.reset(start, end)

            # Track forward
            batch_tractogram, reward = self._validation_episode(
                state, env)

            # Flip streamlines to initialize backwards tracking
            streamlines = [s[::-1] for s in batch_tractogram.streamlines]
            state = backward_env.reset(streamlines)
            del batch_tractogram

            # Track backwards
            batch_tractogram, reward = self._validation_episode(
                state, backward_env)

            yield batch_tractogram, reward

    def run_validation(
        self,
        batch_size,
        env: BaseEnv,
        backward_env: BaseEnv,
    ) -> Tuple[Tractogram, float]:
        """
        Call the main loop

        Parameters
        ----------
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        streamlines: Tractogram
            Tractogram containing the tracked streamline
        running_reward: float
            Cummulative training steps reward
        """
        # Switch policy to eval mode so no gradients are computed
        self.policy.eval()

        # Initialize tractogram
        tractogram = None

        # Reward gotten during validation
        cummulative_reward = 0

        for t, r in self.generate_streamlines(batch_size, env, backward_env):
            if tractogram is None:
                tractogram = t
            else:
                tractogram += t
            cummulative_reward += r

        return tractogram, cummulative_reward

    def run_train(
        self,
        env: BaseEnv,
        back_env: BaseEnv,
        batch_size: int = 2**12,  # TODO: Parametrize this
    ) -> Tuple[Tractogram, float, float, float]:
        """
        Call the main training loop forward then backward

        Parameters
        ----------
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm
        back_env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm. Pre-initialized with half-streamlines
        batch_size: int
            How many streamlines to track at the same time

        Returns
        -------
        streamlines: Tractogram
            Tractogram containing the tracked streamline
        actor_loss: float
            Cumulative policy training loss
        critic_loss: float
            Cumulative critic training loss
        running_reward: float
            Cummulative training steps reward
        """

        self.policy.train()

        running_reward = 0
        running_actor_loss = 0
        running_critic_loss = 0
        running_length = 0

        # Track for every seed in the environment
        # Tracking should not be done in batches for every seed
        # as it would mean more training for each epoch

        state = env.nreset(batch_size)

        # Track forward
        batch_tractogram, reward, actor_loss, critic_loss, length = \
            self._episode(state, env)

        # Flip streamlines to initialize backwards tracking
        streamlines = [s[::-1] for s in batch_tractogram.streamlines]
        state = back_env.reset(streamlines)

        # Track backwards
        batch_tractogram, reward, actor_loss, critic_loss, length = \
            self._episode(state, back_env)

        running_reward += reward
        running_actor_loss += actor_loss
        running_critic_loss += critic_loss
        running_length += length

        return (
            batch_tractogram,
            running_actor_loss,
            running_critic_loss,
            running_reward,
            running_length)
