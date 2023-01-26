# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
import flashy


class Solver(flashy.BaseSolver):
    def __init__(self, cfg, model, loaders, optim):
        super().__init__()
        self.h = cfg
        self.model = model
        self.loaders = loaders
        self.optim = optim

        self.register_stateful('model', 'optim')
        self.init_tensorboard()

    def run(self):
        self.logger.info('Log dir: %s', self.folder)
        self.restore()
        self.log_hyperparams(self.h)
        for epoch in range(self.epoch, self.h.epochs + 1):
            self.run_stage("train", self.do_train_valid, train=True)
            self.run_stage("valid", self.do_train_valid, train=False)
            self.commit()

    def get_formatter(self, stage_name: str):
        return flashy.Formatter({
            'acc': '.1%',
            'loss': '.5f',
        })

    def do_train_valid(self, train: bool = True):
        self.logger.info('-' * 80)
        self.logger.info(f'Starting {self.current_stage} stage...')
        loader = self.loaders["train" if train else "valid"]
        lp = self.log_progress(self.current_stage, loader, total=len(loader), updates=self.h.log_updates)
        average = flashy.averager()

        for idx, batch in enumerate(lp):
            img, label = [x.to(self.h.device) for x in batch]
            est = self.model(img)
            loss = F.cross_entropy(est, label)
            acc = (est.argmax(dim=-1).float() == label).float().mean()
            if train:
                loss.backward()
                flashy.distrib.sync_model(self.model)
                self.optim.step()
                self.optim.zero_grad()

            metrics = average({'acc': acc, 'loss': loss})
            lp.update(**metrics)
            if idx == 0:
                self.log_image(self.current_stage, 'sample', img[0])
            if idx > 20:
                break

        metrics = flashy.distrib.average_metrics(metrics, len(loader))
        return metrics
