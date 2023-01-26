# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""For training adversarial losses, we provide an AdversarialLoss wrapper that encapsulate
the training of the adversarial loss. This allows us to keep the main training loop simple and
to encapsulate the complexity of the adversarial loss training inside this utility class.
"""
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

from . import distrib, utils


LossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]


class AdversarialLoss(nn.Module):
    """
    This is an example class for handling adversarial losses without requiring to mess up
    the main training loop. This will not fit all use case and will need inheriting
    to extend to more complex use (i.e. gradient penalty, or feature loss).

    Args:
        adversary: this will be used to estimate the logits given the fake and real samples.
            We use the convention that the output is high for fake sample.
        optimizer: optimizer used for training the given module.
        loss: loss function, by default binary_cross_entropy_with_logits.


    Example of usage:

        adv_loss = AdversarialLoss(module, optimizer, loss)
        for real in loader:
            noise = torch.randn(...)
            fake = model(noise)
            adv_loss.train_adv(fake, real)
            loss = adv_loss(fake)
            loss.backward()
    """
    def __init__(self, adversary: nn.Module, optimizer: torch.optim.Optimizer,
                 loss: LossType = F.binary_cross_entropy_with_logits):
        super().__init__()
        self.adversary = adversary
        distrib.broadcast_model(adversary)
        self.optimizer = optimizer
        self.loss = loss

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Add the optimizer state dict inside our own.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'optimizer'] = self.optimizer.state_dict()
        return destination

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Load optimizer state.
        self.optimizer.load_state_dict(state_dict.pop(prefix + 'optimizer'))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def train_adv(self, fake: torch.Tensor, real: torch.Tensor):
        """Train the adversary with the given fake and real example.
        This will automatically synchronize gradients (with `flashy.distrib.eager_sync_grad`)
        and call the optimizer.
        """

        logit_fake_is_fake = self.adversary(fake.detach())
        logit_real_is_fake = self.adversary(real.detach())
        one = torch.tensor(1., device=fake.device).expand_as(logit_fake_is_fake)
        zero = torch.tensor(0., device=fake.device).expand_as(logit_real_is_fake)
        loss = self.loss(logit_fake_is_fake, one) + self.loss(logit_real_is_fake, zero)

        self.optimizer.zero_grad()
        with distrib.eager_sync_model(self.adversary):
            loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, fake: torch.Tensor):
        """Return the loss for the generator, i.e. trying to fool the adversary.
        """
        with utils.readonly(self.adversary):
            logit_fake_is_fake = self.adversary(fake)
            zero = torch.tensor(0., device=fake.device).expand_as(logit_fake_is_fake)
            loss_generator = self.loss(logit_fake_is_fake, zero)
        return loss_generator
