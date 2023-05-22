import numpy as np
import torch

from collections import defaultdict
from torch.distributions import Normal, kl_divergence
from typing import Tuple

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.algorithms.shared.onpolicy import ActorCritic
from TrackToLearn.algorithms.shared.replay import ReplayBuffer
from TrackToLearn.algorithms.shared.utils import (
    add_item_to_means, mean_losses)


# From ikostrikov's impl
def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

        flat_params = torch.cat(params)
    return flat_params


# From ikostrikov's impl
def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grads(loss, params, create_graph=False, retain_graph=True):
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, retain_graph=retain_graph)
    flat_grads = torch.cat([grad.view(-1) for grad in grads])
    return flat_grads


# TODO : ADD TYPES AND DESCRIPTION


class TRPO(A2C):
    """
    The sample-gathering and training algorithm.
    Based on:

        Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015, June).
        Trust region policy optimization. In International conference on machine
        learning (pp. 1889-1897). PMLR.

    Implementation is based on
    - https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/trpo/trpo.py # noqa E501
    - https://github.com/ikostrikov/pytorch-trpo/blob/master/trpo.py
    - https://github.com/ajlangley/trpo-pytorch
    - https://github.com/mjacar/pytorch-trpo/blob/master/trpo_agent.py

    Some alterations have been made to the algorithms so it could be fitted to the
    tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_dims: int,
        action_std: float = 0.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lmbda: float = 0.99,
        entropy_loss_coeff: float = 0.01,
        delta: float = 0.01,
        max_backtracks: int = 10,
        backtrack_coeff: float = 0.05,
        K_epochs: int = 1,
        max_traj_length: int = 1,
        n_actors: int = 4096,
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
        hidden_dims: str
            Widths and layers of the NNs
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        lmbda: float
            Lambda parameter for Generalized Advantage Estimation (GAE):
            John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan:
            “High-Dimensional Continuous Control Using Generalized
             Advantage Estimation”, 2015;
            http://arxiv.org/abs/1506.02438 arXiv:1506.02438
        entropy_loss_coeff: float
            Entropy bonus for the actor loss
        delta: float
            Hyperparameter for KFAC. Controls the "distance" between
            the new and old policies.
        max_backtracks: int
            Maximum number of steps to do during line search
        backtrack_coeff: float
            Size of step during line search
        max_traj_length: int
            Maximum trajectory length to store in memory.
        K_epochs: int
            Number of times to update on the same batch.
        n_actors: int
            Number of learners
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
        """

        self.input_size = input_size
        self.action_size = action_size

        self.lr = lr
        self.gamma = gamma

        self.on_policy = True

        # Declare policy
        self.policy = ActorCritic(
            input_size, action_size, hidden_dims, device,
        ).to(device)

        # Note the optimizer is ran on the target network's params
        # TRPO: TRPO may use LGBFS optimization for the value function.
        # Kinda special to TRPO
        # self.optimizer = torch.optim.LBFGS(
        #     self.policy.critic.parameters(), lr=lr, max_iter=25)

        self.optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr)

        # TRPO Specific parameters
        self.lmbda = lmbda
        self.entropy_loss_coeff = entropy_loss_coeff
        self.max_backtracks = max_backtracks

        self.backtrack_coeff = backtrack_coeff
        self.damping = 0.01
        self.delta = delta

        self.max_traj_length = max_traj_length
        self.K_epochs = K_epochs

        self.max_action = 1.
        self.t = 1
        self.device = device
        self.n_actors = n_actors

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size, n_actors,
            max_traj_length, self.gamma, self.lmbda)

        self.rng = rng

    def update(
        self,
        replay_buffer,
        batch_size=8192,
    ) -> Tuple[float, float]:
        """
        Policy update function, where we want to maximize the probability of
        good actions and minimize the probability of bad actions

        The general idea is to compare the current policy and the target
        policies. To do so, the "ratio" is calculated by comparing the
        probabilities of actions for both policies. The ratio is then
        multiplied by the "advantage", which is how better than average
        the policy performs.

        Therefore:
            - actions with a high probability and positive advantage will
              be made a lot more likely
            - actions with a low probabiliy and positive advantage will be made
              more likely
            - actions with a high probability and negative advantage will be
              made a lot less likely
            - actions with a low probabiliy and negative advantage will be made
              less likely

        TRPO adds a twist to this where, since the advantage estimation is done
        with your (potentially bad) networks, a "pessimistic view" is used
        where gains will be clamped, so that high gradients (for very probable
        or with a high-amplitude advantage) are tamed. This is to prevent your
        network from diverging too much in the early stages

        Parameters
        ----------
        replay_buffer: ReplayBuffer
            Replay buffer that contains transitions

        Returns
        -------
        losses: dict
            Dict. containing losses and training-related metrics.
        """

        # Sample replay buffer
        s, a, ret, adv, p, mu, std = \
            replay_buffer.sample()

        running_losses = defaultdict(list)

        for i in range(0, len(s), batch_size):
            j = i + batch_size

            state = torch.FloatTensor(s[i:j]).to(self.device)
            action = torch.FloatTensor(a[i:j]).to(self.device)
            returns = torch.FloatTensor(ret[i:j]).to(self.device)
            advantage = torch.FloatTensor(adv[i:j]).to(self.device)
            old_prob = torch.FloatTensor(p[i:j]).to(self.device)
            old_mu = torch.FloatTensor(mu[i:j]).to(self.device)
            old_std = torch.FloatTensor(std[i:j]).to(self.device)

            # Here be dragons

            def get_kl():

                def kl(mu, std):
                    return kl_divergence(
                        Normal(old_mu.detach(), old_std.detach()),
                        Normal(mu, std)).mean()

                return kl

            def get_loss():
                def loss(policy):
                    _, logprob, entropy, mu, std = policy.evaluate(
                        state,
                        action)

                    ratio = torch.exp(logprob - old_prob)
                    policy_loss = (-advantage * ratio).mean()
                    # TRPO "pessimistic" policy loss
                    # Entropy "loss" to promote entropy in the policy
                    entropy_loss = -self.entropy_loss_coeff * entropy.mean()
                    actor_loss = policy_loss + entropy_loss
                    return actor_loss, mu, std, entropy_loss

                return loss

            def get_hessian(kl):
                """ Compute Hx, in a flattened version
                x is the grad of the actor loss
                """

                flat_grad_kl = get_flat_grads(
                    kl, self.policy.actor.parameters(), create_graph=True)

                def Hx(x):

                    kl_v = (flat_grad_kl @ x.clone())
                    flat_grad_grad_kl = get_flat_grads(
                        kl_v, self.policy.actor.parameters())

                    return flat_grad_grad_kl.detach() + (self.damping * x)

                return Hx

            def compute_conjugate_gradients(b, Hx, nsteps=10):
                """ Compute conjugate gradient of the actor loss gradient
                https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm # noqaE501
                """
                x = torch.zeros(b.size(), device=self.device)
                p = b.clone()
                r = b.clone()  # - Ax, but Ax = 0 with x = 0
                rr = torch.dot(r, r)
                for i in range(nsteps):
                    Ap = Hx(p)
                    alpha = rr / (torch.dot(p, Ap) + 1e-8)
                    x += alpha * p
                    r -= alpha * Ap
                    rr_p = torch.dot(r, r)
                    if rr_p < 1e-10:
                        break
                    p = r + (rr_p / rr) * p
                    rr = rr_p
                return x

            def get_step(g, Hx, delta):
                return torch.sqrt(2 * delta / torch.matmul(g, Hx(g)))

            def linesearch(step, kl, old_params, old_loss):
                # to start backtrack at 1.
                step_size = 1. / self.backtrack_coeff

                for i in np.arange(self.max_backtracks):
                    step_size *= self.backtrack_coeff
                    new_params = old_params + (step * step_size)
                    set_flat_params_to(self.policy.actor, new_params)
                    with torch.no_grad():
                        pi_loss, mu, std, entropy = loss_fn(self.policy)
                        kl_mean = kl(mu, std)
                    expected_improve = expected * step_size
                    actual_improvement = old_loss - pi_loss
                    ratio = actual_improvement / expected_improve

                    # set_flat_params_to(self.policy.actor, old_params)
                    kl_cond = kl_mean <= self.delta
                    ratio_cond = ratio > 0.1
                    improve_cond = actual_improvement > 0.
                    if kl_cond and ratio_cond and improve_cond:
                        # print('Found suitable step', step_size)
                        # print('Improv', ratio)
                        return pi_loss, step_size, kl_mean, entropy
                print('Linesearch failed', ratio, kl_mean)
                return old_loss, step_size, kl_mean, entropy

            loss_fn = get_loss()

            actor_loss, old_mu, old_std, entropy = loss_fn(self.policy)
            kl = get_kl()
            kl_mean = kl(old_mu, old_std)
            loss_grad = get_flat_grads(
                actor_loss, self.policy.actor.parameters())

            Hx = get_hessian(kl_mean)

            # OpenAI baseline update
            step = compute_conjugate_gradients(-loss_grad, Hx)
            max_step_coeff = (2 * self.delta / (step @ Hx(step)))**(0.5)
            max_trpo_step = max_step_coeff * step

            # shs = 0.5 * torch.matmul(g, Hx(g))
            # lm = torch.sqrt(shs / self.delta)
            # max_step = g / lm

            expected = -loss_grad @  max_trpo_step

            old_params = get_flat_params_from(self.policy.actor)
            actor_loss, step_size, kl_mean, entropy = linesearch(
                max_trpo_step, kl, old_params, actor_loss)

            set_flat_params_to(
                self.policy.actor, old_params + (max_trpo_step * step_size))

            # TODO?: Iterate on all data before K ?
            for _ in range(self.K_epochs):

                # To use with LGBFS

                # def critic_step():
                #     # V_pi'(s) and pi'(a|s)
                #     v_s, *_ = self.policy.evaluate(
                #         state,
                #         action)
                #     # TRPO critic loss
                #     critic_loss = ((returns - v_s) ** 2).mean()
                #     # Critic gradient step
                #     self.optimizer.zero_grad()
                #     critic_loss.backward()
                #     return critic_loss

                # self.optimizer.step(critic_step)

                v, *_ = self.policy.evaluate(
                    state,
                    action)

                # TRPO critic loss
                critic_loss = ((returns - v) ** 2).mean()

                # Critic gradient step
                self.optimizer.zero_grad()
                critic_loss.backward()

                self.optimizer.step()

            # TODO: Better loss and metric logging
            losses = {'actor_loss': actor_loss.item(),
                      'critic_loss': critic_loss.item(),
                      'advantage': advantage.mean().item(),
                      'step_size': step_size,
                      'max_trpo_step': max_trpo_step.mean().item(),
                      'returns': returns.mean().item(),
                      'adv': advantage.mean().item(),
                      'v': v.mean().item(),
                      'entropy': entropy.item(),
                      'ret': returns.mean().item(),
                      'kl_mean': kl_mean.item()}

            running_losses = add_item_to_means(running_losses, losses)

        return mean_losses(running_losses)
