# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from dora import hydra_main
import flashy


class Solver(flashy.BaseSolver):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = torch.nn.Linear(32, 1)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.best_state = {}
        # register_stateful supports any attribute. On checkpoints loading,
        # it will try to use inplace method when possible (i.e. Modules, lists, dicts).
        self.register_stateful('model', 'optim', 'best_state')
        self.init_tensorboard()  # all metrics will be reported to stderr and tensorboard.

    def run(self):
        self.restore()  # load checkpoint
        for epoch in range(self.epoch, self.cfg.epochs + 1):
            # Stages are used for automatic metric reporting to Dora, and it also
            # allows tuning how metrics are formatted.
            self.run_stage('train', self.train)
            # Commit will send the metrics to Dora and save checkpoints by default.
            self.commit(save_checkpoint=epoch % 2 == 1)

    def train(self):
        # this is super dumb, checkout `examples/cifar/solver.py` for more advance usage!
        x = torch.randn(4, 32)
        y = self.model(x)
        loss = y.abs().mean()
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return {'loss': loss.item()}


@hydra_main(config_path='config', config_name='config', version_base='1.1')
def main(cfg):
    # Setup logging both to XP specific folder, and to stderr.
    flashy.setup_logging()
    # Initialize distributed training, no need to specify anything when using Dora.
    flashy.distrib.init()
    solver = Solver(cfg)
    solver.run()


if __name__ == '__main__':
    main()
