# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import torch
from torch import nn
from dora import hydra_main

import flashy
from flashy import distrib


class Network(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim
        self.model = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor):
        return self.model(x)


class NoiseDataset:
    def __init__(self, size: int = 10, dim: int = 8):
        self.size = size
        self.dim = dim

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.randn(self.dim)


class Solver(flashy.BaseSolver):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg
        self.teacher = Network(self.h.dim).to(self.h.device)
        distrib.broadcast_model(self.teacher)

        for p in self.teacher.parameters():
            p.requires_grad_(True)
        self.model = Network(self.h.dim).to(self.h.device)
        distrib.broadcast_model(self.model)

        self.optim = torch.optim.Adam(self.model.parameters())

        adv_model = Network(self.h.dim).to(self.h.device)
        adv_opt = torch.optim.Adam(adv_model.parameters())
        self.adv = flashy.adversarial.AdversarialLoss(adv_model, adv_opt)

        self.loader = distrib.loader(
            NoiseDataset(self.h.dset_size, self.h.dim), shuffle=True,
            batch_size=self.h.batch_size, num_workers=self.h.num_workers)

        self.register_stateful('teacher', 'model', 'optim', 'adv')

    def run(self):
        self.logger.info('Log dir: %s', self.folder)
        self.restore()
        for epoch in range(self.epoch, self.h.epochs + 1):
            self.run_stage("train", self.do_train_valid, train=True)
            self.run_stage("valid", self.do_train_valid, train=False)
            self.commit()
            if epoch == self.h.stop_at:
                return

    def get_formatter(self, stage_name: str):
        return flashy.Formatter({
            'loss': '.4f',
            'mse': '.4f',
            'adv_gen': '.4f',
            'adv_disc': '.4f',
        }, exclude_keys=['*'])

    def do_train_valid(self, train: bool = True):
        label = "train" if train else "valid"
        self.logger.info('-' * 80)
        self.logger.info(f'Starting {label} stage...')
        lp = self.log_progress(label, self.loader, updates=self.h.log_updates)
        average = flashy.averager()

        for noise in lp:
            noise = noise.to(self.h.device)
            estimate = self.model(noise)
            gt = self.teacher(noise)
            mse = nn.functional.mse_loss(estimate, gt)
            adv_disc = self.adv.train_adv(estimate, gt)
            adv_gen = self.adv(estimate)
            loss = mse + adv_gen
            if train:
                self.optim.zero_grad()
                loss.backward()
                flashy.distrib.sync_model(self.model)
                self.optim.step()
            metrics = average({'loss': loss, 'mse': mse, 'adv_disc': adv_disc, 'adv_gen': adv_gen})
            lp.update(**metrics)
        metrics = flashy.distrib.average_metrics(metrics, len(self.loader))
        return metrics


@hydra_main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    flashy.setup_logging()
    distrib.init(cfg.distrib_method)
    torch.manual_seed(1234)

    solver = Solver(cfg)
    solver.run()


if '_FLASHY_TMDIR' in os.environ:
    main.dora.dir = os.environ['_FLASHY_TMDIR']
