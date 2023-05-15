from os.path import join as pjoin

from comet_ml import Experiment


class CometMonitor():
    """ Wrapper class to track information using Comet.ml
    """

    def __init__(
        self,
        experiment: Experiment,
        experiment_id: str,
        experiment_path: str,
        prefix: str,
        render: bool = False,
        use_comet: bool = False
    ):
        """
        Parameters:
        -----------
            experiment: str
                Name of experiment. Will contain many interations
                of experiment based on different parameters
            experiment_id: str
                Actual experiment_id of iteration of experiment. Most likey
                datetime it was started at
            experiment_path: str
                Experiment path used to fetch images or other stuff
            prefix: str
                Prefix for metrics
            use_comet: bool
                Whether to actually use comet or not. Useful when
                Comet access is limited
        """
        self.experiment_path = experiment_path
        self.experiment_id = experiment_id
        # IMPORTANT
        # This presumes that your API key is in your home folder or at
        # the project root.
        self.e = experiment
        self.e.add_tag(experiment_id)
        self.prefix = prefix
        self.render = render

    def log_parameters(self, hyperparameters: dict):
        self.e.log_parameters(hyperparameters)

    def update_pretrain(
        self,
        pretrain_actor_monitor,
        pretrain_critic_monitor=None,
        i_episode=0
    ):
        pass

    def update(
        self,
        reward_monitor,
        len_monitor,
        vc_monitor=None,
        ic_monitor=None,
        nc_monitor=None,
        vb_monitor=None,
        ib_monitor=None,
        ol_monitor=None,
        i_episode=0
    ):

        reward_x, reward_y = zip(*reward_monitor.epochs)
        len_x, len_y = zip(*len_monitor.epochs)

        self.e.log_metrics(
            {
                self.prefix + "Reward": reward_y[-1],
                self.prefix + "Length": len_y[-1],
            },
            step=i_episode
        )

        if vc_monitor is not None and len(vc_monitor) > 0:
            vc_x, vc_y = zip(*vc_monitor.epochs)
            nc_x, nc_y = zip(*nc_monitor.epochs)
            ic_x, ic_y = zip(*ic_monitor.epochs)
            vb_x, vb_y = zip(*vb_monitor.epochs)
            ib_x, ib_y = zip(*ib_monitor.epochs)
            ol_x, ol_y = zip(*ol_monitor.epochs)

            self.e.log_metrics(
                {
                    self.prefix + "VC": vc_y[-1],
                    self.prefix + "NC": nc_y[-1],
                    self.prefix + "IC": ic_y[-1],
                    self.prefix + "VB": vb_y[-1],
                    self.prefix + "IB": ib_y[-1],
                    self.prefix + "OL": ol_y[-1],
                },
                step=i_episode
            )

        if self.render:
            self.e.log_image(
                pjoin(self.experiment_path, 'render',
                      '{}.png'.format(i_episode)),
                step=i_episode)

    def log_losses(self, loss_dict, i):
        self.e.log_metrics(loss_dict, step=i)

    def update_train(
        self,
        reward_monitor,
        i_episode,
    ):
        reward_x, reward_y = zip(*reward_monitor.epochs)

        self.e.log_metrics(
            {
                self.prefix + "Train Reward": reward_y[-1],

            },
            step=i_episode
        )
