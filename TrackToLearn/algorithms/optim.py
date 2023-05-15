import numpy as np
import torch


class KFACOptimizer(torch.optim.Optimizer):
    """
    Implementation is based on
     - https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/kfac.py # noqa E501
     - https://github.com/alecwangcq/KFAC-Pytorch/blob/master/optimizers/kfac.py

     See https://www.youtube.com/watch?v=qAVZd6dHxPA for a nice explanation

     I have renamed variables, added commments and references and rearanged functions in
     a way that made more sense to me.

   References:
       [1] - Martens, J., & Grosse, R. (2015, June). Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning (pp. 2408-2417). PMLR. # noqa E501
       [2] - Grosse, R., & Martens, J. (2016, June). A kronecker-factored approximate fisher matrix for convolution layers. In International Conference on Machine Learning (pp. 573-582). PMLR.  # noqa E501
       [3] - Wu, Y., Mansimov, E., Liao, S., Grosse, R., & Ba, J. (2017). Scalable trust-region method for deep reinforcement learning using kronecker-factored approximation. arXiv preprint arXiv:1708.05144.  # noqa E501
     """

    def __init__(self,
                 model,
                 lr=0.25,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 Ts=1,
                 Tf=10):

        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.optim = torch.optim.SGD(
                        model.parameters(),
                        lr=lr * (1 - momentum),
                        momentum=momentum)

        self.known_modules = {'Linear'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        # a = activation of module, referred to as uppercase Gamma in [2]
        # m_aa = Psi/Omega* = second moment matrix of a
        # s = pre-activation derivatives w.r.t loss,
        # m_ss = Gamma = second moment matrix of s
        # See page 5 of [2]
        # d, Q = orthogonal eigendecompositions of Gamma/Psi
        # See page 26 of [2]
        self.m_aa, self.m_ss, self.Q_a, self.Q_s, self.d_a, self.d_s = \
            {}, {}, {}, {}, {}, {}

        self.momentum = momentum
        self.stat_decay = stat_decay

        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.Ts = Ts  # statistics update period
        self.Tf = Tf  # inverse update period

        # I'm confused regarding the difference between Omega and Psi in [2]
        # as they're both referring to the matrix of autocovariance of s

    @staticmethod
    def _get_gradient(m, classname):
        """ Get gradient from layer and handle bias
        """
        p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat(
                [p_grad_mat, m.bias.grad.data.view(-1, 1)], dim=1)
        return p_grad_mat

    @staticmethod
    def _compute_cov_a(a):
        """ Compute Omega/Psi/A. See page 11 of [2]
        """
        batch_size = a.size()[0]
        a = torch.cat([a, a.new_ones(a.size()[0], 1)], dim=1)
        return a.t() @ (a / batch_size)

    @staticmethod
    def _compute_cov_s(s):
        """ Compute Gamma/S. See page 11 of [2]
        """
        batch_size = s.size()[0]
        s_ = s * batch_size
        return s_.t() @ (s_ / batch_size)

    @staticmethod
    def _update_running_stat(m, M, stat_decay):
        """ Update statistic S or A with moving average
        """
        M *= stat_decay / (1 - stat_decay)
        M += m
        M *= (1 - stat_decay)

    def _save_input(self, module, inpt):
        """ Build the Omega/Psi matrix
        """
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            aa = self._compute_cov_a(inpt[0].data)
            if self.steps == 0:
                self.m_aa[module] = aa.clone()
            self._update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        """ Build the Gamma matrix
        """
        if self.acc_stats and self.steps % self.Ts == 0:
            ss = self._compute_cov_s(
                grad_output[0].data)
            # Initialize buffers
            if self.steps == 0:
                self.m_ss[module] = ss.clone()
            self._update_running_stat(ss, self.m_ss[module], self.stat_decay)

    def _prepare_model(self):
        """ Register hooks
        """
        count = 0
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                count += 1

    def _update_inv(self, m):
        """Eigen decomposition for computing inverse of the fisher matrix.
        Assigns the decomposition to self directly. See [2], p.26

        Arguments
        ---------
        m: layer

        Returns
        -------
        None
        """

        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = torch.linalg.eigh(
            self.m_aa[m])
        self.d_s[m], self.Q_s[m] = torch.linalg.eigh(
            self.m_ss[m])

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_s[m].mul_((self.d_s[m] > eps).float())

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """ Compute natural gradient with the trick defined
        in page 26 of [2].

        Arguments
        ---------
        m: layer
        p_grad_mat: gradient matrix
        damping: damping parameter (gamma)

        Returns
        -------
        v: list of gradients w.r.t to the parameters in `m`

        """

        v1 = self.Q_s[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_s[m].unsqueeze(1) *
                   self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_s[m] @ v2 @ self.Q_a[m].t()
        if m.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip(self, updates, lr):
        """ Return clipped update
        """
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, np.sqrt(self.kl_clip / (vg_sum + 1e-10)))
        return nu

    def _update_grad(self, updates, nu):
        """ Update the gradients
        """
        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self):
        """ Apply gradients to weights
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.Ts:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                                torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-lr)

    def step(self, closure=None):
        """ Perform optimizer step
        """

        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.Tf == 0:
                self._update_inv(m)
            p_grad_mat = self._get_gradient(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        nu = self._kl_clip(updates, lr)
        self._update_grad(updates, nu)

        self.optim.step()
        # self._step()
        self.steps += 1
