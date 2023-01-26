# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dora import hydra_main
import flashy
import torch
import torchvision
from torchvision import models, transforms


from .solver import Solver


def get_datasets(root: str):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_cv = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    tr_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    cv_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_cv)

    return tr_set, cv_set


def get_solver(cfg):
    bs = cfg.optim.batch_size
    tr_set, cv_set = get_datasets(cfg.data.root)
    tr_loader = flashy.distrib.loader(tr_set, batch_size=bs)
    cv_loader = flashy.distrib.loader(cv_set, batch_size=bs)
    loaders = {'train': tr_loader, 'valid': cv_loader}
    model = models.resnet18(num_classes=10).to(cfg.device)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.optim.lr)
    return Solver(cfg, model, loaders, optim)


def get_solver_from_sig(sig: str):
    xp = main.get_xp_from_sig(sig)
    with xp.enter():
        solver = get_solver(xp.cfg)
    solver.restore()
    return solver


@hydra_main(config_path='config', config_name='config', version_base='1.1')
def main(cfg):
    flashy.setup_logging()
    flashy.distrib.init()
    solver = get_solver(cfg)
    solver.run()


if __name__ == '__main__':
    main()
