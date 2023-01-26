# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import typing as tp
import warnings

import dora
import torch

try:
    import wandb
except ModuleNotFoundError:
    wandb = None  # type: ignore

from . import ExperimentLogger
from .utils import (
    _add_prefix, _convert_params, _flatten_dict,
    _sanitize_params
)
from ..distrib import rank_zero_only, is_rank_zero


class WandbLogger(ExperimentLogger):
    """ExperimentLogger for Wandb (Weight and Biases)

    Args:
        save_dir (str): The directory where the experiment logs are written to
        with_media_logging (bool): Whether to save media samples with the logger or ignore them
            This comes at bandwidth costs for Wandb but we are usually fine with it
        project (str): Wandb project name
        id (Optional[str]): Wandb run id
        group (Optional[str]): Wandb group
        reinit (bool): Whether to reinit run or not
        resume (bool): Whether to resume run or not
        name (str): Name for the experiment logs
        kwargs (Any): Additional wandb parameters, including tags for example
    """
    def __init__(self, save_dir: str, with_media_logging: bool = True, project: tp.Optional[str] = None,
                 id: tp.Optional[str] = None, group: tp.Optional[str] = None, reinit: bool = False,
                 resume: tp.Union[str, bool] = False, name: tp.Optional[str] = None, **kwargs: tp.Any):
        self._save_dir = save_dir
        self._with_media_logging = with_media_logging
        self._name = name or 'wandb'
        self._project = project
        self._id = id
        self._group = group
        self._reinit = reinit
        self._resume = resume
        self._wandb_run: tp.Optional[tp.Any] = None
        self._init_wandb(**kwargs)

    @rank_zero_only
    def _init_wandb(self, **kwargs):
        if wandb:
            self._wandb_run = wandb.init(
                project=self._project,
                reinit=self._reinit,
                group=self._group,
                name=self._name,
                dir=self.save_dir,
                id=self._id,
                resume=self._resume,
                **kwargs
            )
        else:
            warnings.warn("wandb package was not found: use pip install wandb")

    @property  # type: ignore
    @rank_zero_only
    def writer(self) -> tp.Optional[tp.Any]:
        """Actual wandb run object."""
        return self._wandb_run

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

        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_params(params)
        self.writer.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, prefix: tp.Union[str, tp.List[str]], metrics: dict, step: tp.Optional[int] = None) -> None:
        """Records metrics.
        This method logs metrics as as soon as it received them.

        Args:
            prefix: Prefix(es) to use for metric names when writing to smart logger
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        assert is_rank_zero(), "writer tried to log from global_rank != 0"
        if self.is_disabled() or not self.with_media_logging:
            return

        metrics = _add_prefix(metrics, prefix, self.group_separator)

        for key, val in metrics.items():
            wandb.log({key: val}, step=step)

    @rank_zero_only
    def log_audio(self, prefix: tp.Union[str, tp.List[str]], key: str, audio: tp.Any, sample_rate: int,
                  step: tp.Optional[int] = None, **kwargs: tp.Any) -> None:
        """Records audio.
        This method logs audio wave as soon as it received them.

        Args:
            prefix: Prefix(es) to use for metric names when writing to smart logger
            key: Key for the audio
            audio: Torch Tensor representing the audio as [C, T]
            step: Step number at which the metrics should be recorded
        """
        assert is_rank_zero(), "writer tried to log from global_rank != 0"
        if self.is_disabled() or not self.with_media_logging:
            return

        assert isinstance(audio, torch.Tensor), "Only support logging torch.Tensor as audio"

        audio = audio.t().clamp(-0.99, 0.99).numpy()
        metrics = {
            key: wandb.Audio(audio, sample_rate=sample_rate, **kwargs)
        }
        metrics = _add_prefix(metrics, prefix, self.group_separator)
        self.log_metrics(metrics, prefix, step)

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
        assert is_rank_zero(), "writer tried to log from global_rank != 0"
        if self.is_disabled() or not self.with_media_logging:
            return

        assert isinstance(image, torch.Tensor), "Only support logging torch.Tensor as image"

        metrics = {
            key: wandb.Image(image, **kwargs)
        }
        metrics = _add_prefix(metrics, prefix, self.group_separator)
        self.log_metrics(metrics, prefix, step)

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
            key: wandb.Table(columns=[key], data=[text], **kwargs)
        }
        metrics = _add_prefix(metrics, prefix, self.group_separator)
        self.log_metrics(metrics, prefix, step)

    @property
    def with_media_logging(self) -> bool:
        """Whether the logger can save media or ignore them."""
        return self._with_media_logging

    @property
    def save_dir(self):
        """Directory where the data is saved."""
        return self._save_dir

    @property
    def name(self):
        """Name of the experiment logger."""
        return self._name

    @classmethod
    def from_xp(cls, with_media_logging: bool = True, project: tp.Optional[str] = None, name: tp.Optional[str] = None,
                group: tp.Optional[str] = None, **kwargs: tp.Any):
        xp = dora.get_xp()
        save_dir = xp.folder
        save_dir.mkdir(exist_ok=True)
        flag_file = xp.folder / 'wandb_flag'
        resume = flag_file.exists()
        flag_file.touch()
        config = None
        if wandb:
            api = wandb.Api()
            try:
                if project:
                    run = api.run(project + '/' + xp.sig)
                else:
                    run = api.run(xp.sig)
            except wandb.CommError:
                pass
            else:
                group = run.group
                name = run.name
                config = run.config
        return WandbLogger(str(save_dir), with_media_logging, id=xp.sig, name=name or xp.sig, group=group,
                           config=config, project=project, resume='allow' if resume else False, **kwargs)
