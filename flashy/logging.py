# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Logging related utilities.
"""
from argparse import Namespace
from collections.abc import Iterable, Sized
import logging
from pathlib import Path
import sys
import time
import typing as tp

import colorlog
from dora import get_xp
from dora.distrib import get_distrib_spec
from flashy.loggers.base import ExperimentLogger
from flashy.loggers.localfs import LocalFSLogger

from .formatter import Formatter
from .loggers import TensorboardLogger, WandbLogger
from .utils import AnyPath


def setup_logging(
        with_file_log: bool = True,
        folder: tp.Optional[AnyPath] = None,
        log_name: str = 'solver.log.{rank}',
        level: int = logging.INFO):
    """Setup logging nicely, we recommend you call this as the very first step,
    not to miss any possible messages. By default this will also create a log file in the experiment folder.

    Args:
        with_file_log: if True, creates a log file in the XP folder,
            or the folder given explicitely. Default is True.
        folder: customize folder to store the logs in.
        log_name: template for the filename of the log. Default is
            `solver.log.{rank}`.
        level: log level, default is `logging.INFO`.
    """
    # See https://docs.python.org/3/howto/logging-cookbook.html#logging-cookbook for reference.
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)

    # Let us switch to colorlog for an improved esthetic experience.
    log_format = ('[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]'
                  '[%(log_color)s%(levelname)s%(reset)s] - %(message)s')
    formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt="%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)

    if with_file_log:
        if folder is None:
            folder = get_xp().folder
        # We need to get the rank in a reliable way, even if distributed is not yet initialized.
        rank = get_distrib_spec().rank
        fh = logging.FileHandler(Path(folder) / log_name.format(rank=rank))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)


def colorize(text: str, color: str) -> str:
    """ANSI colorization with ANSI escape sequence.
    See: https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences

    Args:
        text (str): text to colorize
        color (str): ANSI color escape sequence
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text: str) -> str:
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")


class LogProgressBar:
    """Log progress bar for results
    Sort of like tqdm but using log lines and not as real time.

    Args:
        logger (logging.Logger): Logger obtained from `logging.getLogger`
        iterable (Iterable): Iterable object to wrap
        updates (int): Number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length
        time_per_it (bool): Force speed to display as ms/it
        total (int): length of the iterable, in case it does not support
            `len`.
        name (str): Prefix to use in the log
        level (int): Logging level (like `logging.INFO`)
        delimiter (str): Delimiter between displayed stats in logs
        items_delimiter (str): Delimiter between key, value items in metrics log
    """
    def __init__(self,
                 logger: logging.Logger,
                 iterable: Iterable,
                 updates: int = 5,
                 min_interval: int = 1,
                 time_per_it: bool = False,
                 total: tp.Optional[int] = None,
                 name: str = 'LogProgressBar',
                 level: int = logging.INFO,
                 delimiter: str = '|',
                 items_delimiter: str = ' ',
                 formatter: Formatter = Formatter()):
        self._iterable = iterable
        if total is None:
            assert isinstance(iterable, Sized)
            total = len(iterable)
        self._total = total
        self._updates = updates
        self._min_interval = min_interval
        self._time_per_it = time_per_it
        self._name = name
        self._logger = logger
        self._level = level
        self._delimiter = delimiter
        self._items_delimiter = items_delimiter
        self._formatter = formatter

    def update(self, **metrics) -> bool:
        """Update the metrics to show when logging. Return True if logging will
        happen at the end of this iteration."""
        self._metrics = self._formatter(metrics)
        return self._will_log

    def __iter__(self):
        self._iterator = iter(self._iterable)
        self._will_log = False
        self._index = -1
        self._metrics = {}
        self._begin = time.time()
        return self

    def __next__(self):
        if self._will_log:
            self._log()
            self._will_log = False
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            self._index += 1
            if self._updates > 0:
                _log_every = max(self._min_interval, self._total // self._updates)
                # logging is delayed by 1 it, in order to have the metrics from update
                if self._index >= 1 and self._index % _log_every == 0:
                    self._will_log = True
            return value

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = [f"{k}{self._items_delimiter}{v}" for k, v in self._metrics.items()]
        if self._speed < 1e-4:
            speed = 'oo sec/it'
        elif self._time_per_it and self._speed < 1:
            speed = f'{1 / self._speed:.2f} sec/it'
        elif self._time_per_it:
            speed = f'{1000 / self._speed:.1f} ms/it'
        elif self._speed < 0.1:
            speed = f'{1/self._speed:.1f} sec/it'
        else:
            speed = f'{self._speed:.2f} it/sec'
        prefix = [f'{self._name}', f'{self._index}/{self._total}', f'{speed}']
        msg = f' {self._delimiter} '.join(prefix + infos)
        self._logger.log(self._level, msg)


class ResultLogger:
    """Logger for experiment results.

    Logs summary of training metrics in stdout and media samples
    across the specified platforms experiment loggers.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO,
                 delimiter: str = '|'):
        self._logger = logger
        self._level = level
        self._delimiter = delimiter
        self._experiment_loggers: tp.Dict[str, ExperimentLogger] = {}
        self._experiment_loggers['local'] = LocalFSLogger.from_xp(with_media_logging=True)

    def init_tensorboard(self, **kwargs):
        """Initialize Tensorboard logger using dora xp.
        See `flashy.loggers.tensorboard.TensorboardLogger.from_xp` for details
        """
        self._experiment_loggers['tensorboard'] = TensorboardLogger.from_xp(**kwargs)

    def init_wandb(self, **kwargs):
        """Initialize Wandb logger using dora xp.
        See `flashy.loggers.wandb.WandbLogger.from_xp` for details
        """
        self._experiment_loggers['wandb'] = WandbLogger.from_xp(**kwargs)

    def log_hyperparams(self, params: tp.Union[tp.Dict[str, tp.Any], Namespace],
                        metrics: tp.Optional[dict] = None) -> None:
        """Log hyperparameters to all active experiment loggers.
        See `flashy.loggers.base.ExperimentLogger` for details.
        """
        for _, logger in self._experiment_loggers.items():
            logger.log_hyperparams(params, metrics)

    def get_log_progress_bar(self, stage: str, iterable: Iterable, updates: int = 5, total: tp.Optional[int] = None,
                             step: tp.Optional[int] = None, step_name: tp.Optional[str] = None,
                             **kwargs: tp.Any) -> LogProgressBar:
        """Get a progress bar logger formatted for the current stage and iterable.

        Args:
            stage (str): Stage name
            iterable (Iterable): Data to iterate over
            updates (int): Number of updates for the progress bar logging
            total (int): Total size of the data iterated over
            step (int): Number of current steps
            step_name (str): Name for the used steps (eg. `epochs`)
            kwargs: Additional arguments for :class:`~flashy.logging.LogProgressBar` initialization

        Returns:
            `flashy.logging.LogProgressBar`: Progress bar logger
        """
        name = [f'{stage.capitalize()}']
        if step is not None and step_name is not None:
            name += [f'{step_name.capitalize()} {step}']
        progress_bar_name = f' {self._delimiter} '.join(name)
        return LogProgressBar(self._logger, iterable, updates=updates, total=total,
                              name=progress_bar_name, delimiter=self._delimiter, **kwargs)

    def _log_summary(self, stage: str, metrics: dict,
                     step: tp.Optional[int] = None, step_name: str = "epoch",
                     formatter: Formatter = Formatter()) -> None:
        """Log stage summary of current step with key metrics.

        Args:
            stage (str): Stage name
            metrics (dict): Dictionary of metrics to log (optionally already formatted)
            step (int): Number of current steps
            step_name (str): Name for the used steps (eg. `epochs`)
        """
        out = [f'{stage.capitalize()} Summary']
        if step is not None:
            out += [f'{step_name.capitalize()} {step}']
        metrics = formatter(metrics)
        out += [f"{key}={val}".strip() for key, val in metrics.items()]
        msg = f' {self._delimiter} '.join(out)
        self._logger.log(self._level, bold(msg))

    def log_metrics(self, stage: str, metrics: dict, step: tp.Optional[int] = None,
                    step_name: str = "epoch",
                    formatter: Formatter = Formatter()) -> None:
        """Log metrics to all active experiment loggers.
        See `flashy.loggers.base.ExperimentLogger` for details.
        """
        self._log_summary(stage, metrics, step, step_name, formatter)
        for _, logger in self._experiment_loggers.items():
            logger.log_metrics(stage, metrics, step)

    def log_audio(self, stage: str, key: str, audio: tp.Any, sample_rate: int,
                  step: tp.Optional[int] = None, **kwargs) -> None:
        """Log audio to all active experiment loggers.
        See `flashy.loggers.base.ExperimentLogger` for details.
        """
        for _, logger in self._experiment_loggers.items():
            logger.log_audio(stage, key, audio, sample_rate, step, **kwargs)

    def log_image(self, stage: str, key: str, image: tp.Any,
                  step: tp.Optional[int] = None, **kwargs) -> None:
        """Log image to all active experiment loggers.
        See `flashy.loggers.base.ExperimentLogger` for details.
        """
        for _, logger in self._experiment_loggers.items():
            logger.log_image(stage, key, image, step, **kwargs)

    def log_text(self, stage: str, key: str, text: str, step: tp.Optional[int] = None, **kwargs) -> None:
        """Log text to all active experiment loggers.
        See `flashy.loggers.base.ExperimentLogger` for details.
        """
        for _, logger in self._experiment_loggers.items():
            logger.log_text(stage, key, text, step, **kwargs)
