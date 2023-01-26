# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from pathlib import Path
import typing as tp
import warnings

import dora
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore

from . import ExperimentLogger
from .utils import (
    _add_prefix, _convert_params, _flatten_dict,
    _sanitize_params
)
from ..distrib import rank_zero_only, is_rank_zero


class TensorboardLogger(ExperimentLogger):
    """ExperimentLogger for Tensorboard

    Args:
        save_dir (str): The directory where the experiment logs are written to
        with_media_logging (bool): Whether to save media samples with the logger or ignore them
            This comes at extra storage costs for Tensorboard so we might not want to enforce it
        name (str): Name for the experiment logs
        kwargs (Any): Additional tensorboard parameters
    """
    def __init__(self, save_dir: tp.Union[Path, str], with_media_logging: bool = True,
                 name: tp.Optional[str] = None, **kwargs: tp.Any):
        self._with_media_logging = with_media_logging
        self._save_dir = str(save_dir)
        self._name = name or 'tensorboard'
        self._writer: tp.Optional[SummaryWriter] = None
        if SummaryWriter is not None:
            self._writer = SummaryWriter(self.save_dir, **kwargs)
        else:
            warnings.warn("tensorboard package was not found: use pip install tensorboard")

    @property  # type: ignore
    @rank_zero_only
    def writer(self) -> tp.Optional[SummaryWriter]:
        """Actual tensorboard writer object."""
        return self._writer

    @rank_zero_only
    def is_disabled(self) -> bool:
        return self.writer is None

    @rank_zero_only
    def log_hyperparams(self, params: tp.Union[tp.Dict[str, tp.Any], Namespace],
                        metrics: tp.Optional[dict] = None) -> None:
        """Record experiment hyperparameters.
        This method logs the hyperparameters associated to the experiment.

        Args:
            params: Dictionary of hyperparameters
            metrics: Dictionary of final metrics for the set of hyperparameters
        """
        assert is_rank_zero(), "experiment tried to log from global_rank != 0"
        if self.is_disabled():
            return

        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_params(params)
        if metrics is None or len(metrics) == 0:
            metrics = {'hparams_metrics': -1}
        self.writer.add_hparams(params, metrics)

    @rank_zero_only
    def log_metrics(self, prefix: tp.Union[str, tp.List[str]], metrics: dict, step: tp.Optional[int] = None) -> None:
        """Records metrics.
        This method logs metrics as as soon as it received them.

        Args:
            prefix: Prefix(es) to use for metric names when writing to smart logger
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        assert is_rank_zero(), "experiment tried to log from global_rank != 0"
        if self.is_disabled():
            return

        metrics = _add_prefix(metrics, prefix, self.group_separator)

        for key, val in metrics.items():
            if isinstance(val, torch.Tensor):
                val = val.item()

            if isinstance(val, dict):
                self.writer.add_scalars(key, val, step)
            else:
                try:
                    self.writer.add_scalar(key, val, step)
                except Exception as ex:
                    msg = f"\n you tried to log {val} ({type(val)}) which is currently not supported. " \
                        "Try a dict or a scalar/tensor."
                    raise ValueError(msg) from ex

    @rank_zero_only
    def log_audio(self, key: str, prefix: tp.Union[str, tp.List[str]], audio: tp.Any, sample_rate: int,
                  step: tp.Optional[int] = None, **kwargs: tp.Any) -> None:
        """Records audio.
        This method logs audio wave as soon as it received them.

        Args:
            prefix: Prefix(es) to use for metric names when writing to smart logger
            key: Key for the audio
            audio: Torch Tensor representing the audio as [C, T]
            sample_rate: Sample rate corresponding to the audio
            step: Step number at which the metrics should be recorded
        """
        assert is_rank_zero(), "experiment tried to log from global_rank != 0"
        if self.is_disabled() or not self.with_media_logging:
            return

        assert isinstance(audio, torch.Tensor), "Only support logging torch.Tensor as audio"

        metrics = {
            key: audio.mean(dim=-2, keepdim=True).clamp(-0.99, 0.99)
        }
        metrics = _add_prefix(metrics, prefix, self.group_separator)
        for name, media in metrics.items():
            self.writer.add_audio(
                name,
                media,
                step,
                sample_rate,
                **kwargs
            )

    @rank_zero_only
    def log_image(self, prefix: tp.Union[str, tp.List[str]], key: str, image: tp.Any,
                  step: tp.Optional[int] = None, **kwargs: tp.Any) -> None:
        """Records image.
        This method logs image as soon as it received them.

        Args:
            prefix: Prefix(es) to use for metric names when writing to smart logger
            key: Key for the image
            image: Torch Tensor representing the image
            step: Step number at which the metrics should be recorded
        """
        assert is_rank_zero(), "experiment tried to log from global_rank != 0"
        if self.is_disabled() or not self.with_media_logging:
            return

        assert isinstance(image, torch.Tensor), "Only support logging torch.Tensor as image"

        metrics = {
            key: image
        }
        metrics = _add_prefix(metrics, prefix, self.group_separator)
        for name, media in metrics.items():
            self.writer.add_image(
                name,
                media,
                step,
                **kwargs
            )

    @rank_zero_only
    def log_text(self, prefix: tp.Union[str, tp.List[str]], key: str, text: str,
                 step: tp.Optional[int] = None, **kwargs: tp.Any) -> None:
        """Records text.
        This method logs text as soon as it received them.

        Args:
            prefix: Prefix(es) to use for metric names when writing to smart logger
            key: Key for the text
            text: String containing message
            step: Step number at which the metrics should be recorded
        """
        assert is_rank_zero(), "writer tried to log from global_rank != 0"
        if self.is_disabled() or not self.with_media_logging:
            return

        metrics = {
            key: text
        }
        metrics = _add_prefix(metrics, prefix, self.group_separator)
        for name, media in metrics.items():
            self.writer.add_text(
                name,
                media,
                step,
                **kwargs
            )

    @property
    def save_dir(self) -> str:
        """Directory where the data is saved."""
        return self._save_dir

    @property
    def with_media_logging(self) -> bool:
        """Whether the logger can save media or ignore them."""
        return self._with_media_logging

    @property
    def name(self) -> str:
        """Name of the experiment logger."""
        return self._name

    @classmethod
    def from_xp(cls, with_media_logging: bool = True, name: tp.Optional[str] = None, sub_dir: tp.Optional[str] = None):
        save_dir = dora.get_xp().folder / 'tensorboard'
        if sub_dir:
            save_dir = save_dir / sub_dir
        save_dir.mkdir(exist_ok=True, parents=True)
        return TensorboardLogger(save_dir, with_media_logging, name=name)
