# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from argparse import Namespace
import typing as tp


class ExperimentLogger(ABC):
    """Base interface for logging to experiment management tools"""

    @abstractmethod
    def log_hyperparams(self, params: tp.Union[tp.Dict[str, tp.Any], Namespace],
                        metrics: tp.Optional[dict] = None) -> None:
        """Record experiment hyperparameters.
        This method logs the hyperparameters associated to the experiment.

        Args:
            params: Dictionary of hyperparameters
            metrics: Dictionary of final metrics for the set of hyperparameters
        """
        ...

    @abstractmethod
    def log_metrics(self, prefix: tp.Union[str, tp.List[str]], metrics: dict,
                    step: tp.Optional[int] = None) -> None:
        """Records metrics.
        This method logs metrics as as soon as it received them.

        Args:
            prefix: Prefix(es) to use for metric names when writing to smart logger
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        ...

    @abstractmethod
    def log_audio(self, prefix: tp.Union[str, tp.List[str]], key: str, audio: tp.Any, sample_rate: int,
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
        ...

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @property
    @abstractmethod
    def with_media_logging(self) -> bool:
        """Whether the logger can save media or ignore them."""
        ...

    @property
    @abstractmethod
    def save_dir(self) -> tp.Optional[str]:
        """Directory where the data is saved."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the experiment logger."""
        ...

    @property
    def group_separator(self) -> str:
        """Character used as group separator."""
        return '/'
