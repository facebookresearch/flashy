# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Base Solver class. Specific solver should inherit this class.
Solver takes care of various things, like setting up logging?
As well as running stages.
"""
import logging
from pathlib import Path
import time
import typing as tp

from dora import get_xp
import torch

from .distrib import is_rank_zero
from .formatter import Formatter
from .logging import LogProgressBar, ResultLogger
from .state import StateManager, AttributeWrapper
from .utils import write_and_rename


StageCallable = tp.Callable
logger = logging.getLogger(__name__)


class BaseSolver:
    def __init__(self) -> None:
        self.stateful = StateManager()
        self.xp = get_xp()
        self.register_stateful('history')
        self.register_stateful('xp.cfg', 'xp.sig', write_only=True)
        self.logger = logger
        self.result_logger = ResultLogger(self.logger)

        self._current_stage: tp.Optional[str] = None
        self._current_formatter: tp.Optional[Formatter] = None
        self._start_epoch()

    def _start_epoch(self) -> None:
        self._pending_metrics: tp.Dict[str, tp.Any] = {}

    @property
    def checkpoint_path(self) -> Path:
        return self.folder / 'checkpoint.th'

    @property
    def history(self) -> tp.List[tp.Dict[str, tp.Any]]:
        return self.xp.link.history

    @property
    def folder(self) -> Path:
        return self.xp.folder

    @property
    def epoch(self) -> int:
        return len(self.history) + 1

    def init_tensorboard(self, **kwargs):
        """Initialize Tensorboard logging from Dora xp.
        See `flashy.logging.ResultLogger.init_tensorboard` for details

        Args:
            with_media_logging (bool): Whether to also log media to Tensorboard. Default: True
            name (str): Optional name for the experiment
            sub_dir (str): Optional sub directory in the xp folder to store the logs
        """
        self.result_logger.init_tensorboard(**kwargs)

    def init_wandb(self, **kwargs):
        """Initialize Wandb logging from Dora xp.
        See `flashy.logging.ResultLogger.init_wandb` for details

        Args:
            with_media_logging (bool): Whether to also log media to Wandb. Default: True
            project (str): Optional wandb project name
            name (str): Optional name for the experiment
            group (str): Optional group for the experiment
            kwargs: Additional arguments for :class:`~flashy.loggers.wandb.WandbLogger` initialization
        """
        self.result_logger.init_wandb(**kwargs)

    def _check_in_stage(self):
        if self._current_stage is None:
            raise RuntimeError("This function can only be called from inside a stage.")

    def log_progress(self, stage_name: str, iterable: tp.Iterable,
                     total: tp.Optional[int] = None, updates: int = 5) -> LogProgressBar:
        """See `flashy.logging.ResultLogger.get_log_progress_bar` for details"""
        return self.result_logger.get_log_progress_bar(
            stage_name, iterable, total=total, updates=updates,
            step=self.epoch, step_name='epoch', formatter=self.formatter)

    def log_hyperparams(self, params: dict, metrics: tp.Optional[dict] = None):
        """See `flashy.logging.ResultLogger.log_hyperparams` for details"""
        self.result_logger.log_hyperparams(params, metrics)

    def log_metrics(self, stage_name: str, metrics: dict, formatter: tp.Optional[Formatter] = None):
        """
        Log metrics for a given stage. Note that the overall metrics for a stage ran
        with `run_stage` are automatically logged from the returned dict of metrics.
        You might want however to log other metrics with a different stage name.
        If called from outside a stage, you must pass the Formatter explicitely.

        See `flashy.logging.ResultLogger.log_metrics` for details"""
        if stage_name in self._pending_metrics:
            raise RuntimeError(f"Stage {stage_name} already exist for epoch {self.epoch}")
        self._pending_metrics[stage_name] = metrics
        if formatter is None:
            formatter = self.formatter
        self.result_logger.log_metrics(stage_name, metrics, step=self.epoch, step_name='epoch',
                                       formatter=formatter)

    def log_audio(self, stage_name: str, key: str, audio: tp.Any, sample_rate: int, **kwargs: tp.Any):
        """See `flashy.logging.ResultLogger.log_audio` for details"""
        self.result_logger.log_audio(stage_name, key, audio, sample_rate, self.epoch, **kwargs)

    def log_image(self, stage_name: str, key: str, image: tp.Any, **kwargs: tp.Any):
        """See `flashy.logging.ResultLogger.log_image` for details"""
        self.result_logger.log_image(stage_name, key, image, self.epoch, **kwargs)

    def log_text(self, stage_name: str, key: str, text: str, **kwargs: tp.Any):
        """See `flashy.logging.ResultLogger.log_text` for details"""
        self.result_logger.log_text(stage_name, key, text, self.epoch, **kwargs)

    def register_stateful(self, *args: str, write_only: bool = False):
        """Shortcut around `StateManager.register` method. You can pass any number of
        attribute, included nested attributes and those will be included into the checkpoints
        and automatically restored when `BaseSolver.restore` is called.

        If `write_only` is True, state is only stored and not restored.
        """
        for name in args:
            owner = self
            *path, leaf = name.split(".")
            for part in path:
                owner = getattr(owner, part)
            state_source = AttributeWrapper(owner, leaf)
            self.stateful.register(name, state_source, write_only)

    def state_dict(self):
        return self.stateful.state_dict()

    def load_state_dict(self, state):
        self.stateful.load_state_dict(state)

    def commit(self, save_checkpoint: bool = True):
        self.history.append(self._pending_metrics)
        self._start_epoch()
        if is_rank_zero():
            self.xp.link.update_history(self.history)
            if save_checkpoint:
                state = self.state_dict()
                with write_and_rename(self.checkpoint_path) as f:
                    torch.save(state, f)
                self.logger.debug("Checkpoint saved to %s", self.checkpoint_path)

    def restore(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        state = torch.load(self.checkpoint_path, 'cpu')
        self.load_state_dict(state)
        # TODO: Move to StandardSolver when it exists
        # if len(self.history) > 0:
        #     logger.info("Replaying past metrics...")
        #     for epoch, stages in enumerate(self.history):
        #         for stage_name, metrics in stages.items():
        #             formatted_metrics = self.formatter(metrics)
        #             logger.info("%s", default_format_summary(stage_name, formatted_metrics, epoch))

        self.logger.debug("Checkpoint loaded from %s", self.checkpoint_path)
        return True

    def get_formatter(self, stage_name: str) -> Formatter:
        return Formatter()

    @property
    def formatter(self) -> Formatter:
        self._check_in_stage()
        assert self._current_formatter is not None
        return self._current_formatter

    @property
    def current_stage(self) -> str:
        self._check_in_stage()
        assert self._current_stage is not None
        return self._current_stage

    def run_stage(self, stage_name, method, *args, **kwargs):
        assert self._current_stage is None
        self._current_stage = stage_name
        self._current_formatter = self.get_formatter(stage_name)

        begin = time.time()
        try:
            metrics = method(*args, **kwargs)
            if metrics is None:
                metrics = {}
            metrics["duration"] = time.time() - begin
            self.log_metrics(stage_name, metrics)
        finally:
            self._current_stage = None
            self._current_formatter = None

        return metrics

    def run(self):
        raise NotImplementedError()
