# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
from pathlib import Path
import typing as tp

import dora
import torch

from . import ExperimentLogger
from .utils import (
    _convert_params, _flatten_dict, _fmt_prefix,
    _sanitize_params
)
from ..distrib import rank_zero_only, is_rank_zero


class LocalFSLogger(ExperimentLogger):
    """Local Filesystem logger

    Args:
        save_dir (str): The directory where the experiment logs are written to
        with_media_logging (bool): Whether to save media samples with the logger or ignore them
        name (str): Name for the experiment logs
    """
    def __init__(self, save_dir: tp.Union[Path, str], with_media_logging: bool, name: tp.Optional[str] = None,
                 use_subdirs: bool = False):
        self._save_dir = str(save_dir)
        self._with_media_logging = with_media_logging
        self._name = name or 'local'
        self._use_subdirs = use_subdirs

    def _format_path(self, key: str, suffix: str, prefix: tp.Union[str, tp.List[str]],
                     step: tp.Optional[int] = None) -> Path:
        prefixes = prefix if isinstance(prefix, list) else [prefix]
        if step is not None:
            prefixes.append(str(step))
        subdir = _fmt_prefix(prefixes, self.group_separator)
        path = Path(self.save_dir) / subdir / f'{key}.{suffix}'
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

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
        path = Path(self.save_dir) / 'hyperparams.json'

        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_params(params)

        with open(path, 'w') as fout:
            json.dump(params, fout)

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
        # Don't log anything locally as this is likely to be logged through regular logger

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
        import torchaudio
        assert is_rank_zero(), "experiment tried to log from global_rank != 0"
        if not self.with_media_logging:
            return

        assert isinstance(audio, torch.Tensor), "Only support logging torch.Tensor as audio"

        path = self._format_path(key, 'wav', prefix, step)
        default_kwargs = {'encoding': 'PCM_S', 'bits_per_sample': 16}
        kwargs = {**default_kwargs, **kwargs}
        torchaudio.save(path, audio, sample_rate, **kwargs)

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
        import torchvision
        assert is_rank_zero(), "experiment tried to log from global_rank != 0"
        if not self.with_media_logging:
            return

        assert isinstance(image, torch.Tensor), "Only support logging torch.Tensor as image"
        path = self._format_path(key, 'png', prefix, step)
        torchvision.utils.save_image(image, path, **kwargs)

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
        if not self.with_media_logging:
            return

        path = self._format_path(key, 'txt', prefix, step)
        with open(path, 'w') as fout:
            fout.write(text)

    @property
    def with_media_logging(self) -> bool:
        """Whether the logger can save media or ignore them."""
        return self._with_media_logging

    @property
    def save_dir(self) -> str:
        """Directory where the data is saved."""
        return self._save_dir

    @property
    def name(self) -> str:
        """Name of the experiment logger."""
        return self._name

    @property
    def group_separator(self) -> str:
        """Character used as group separator."""
        return '/' if self._use_subdirs else '_'

    @classmethod
    def from_xp(cls, with_media_logging: bool = True,
                name: tp.Optional[str] = None, sub_dir: tp.Optional[str] = None):
        save_dir = dora.get_xp().folder / 'outputs'
        if sub_dir:
            save_dir = save_dir / sub_dir
        save_dir.mkdir(exist_ok=True, parents=True)
        return LocalFSLogger(save_dir, with_media_logging, name)
