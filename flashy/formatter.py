# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Formatter takes care of formatting metrics for display in the logs.
For each possible training stage, it takes a mapping from metric pattern
to formatting strings
"""
import typing as tp
from fnmatch import fnmatchcase


class Formatter:
    """Define formatting for the file and terminal loggers.
    Most arguments are pattern based, i.e. you can match several metric names
    using shell-like wildcard, for instance `acc_*` for matching all metrics
    starting with `acc_`. Use `__call__` methods on a dict of metrics
    to get relevant formatted metrics.

    Args:
        formats: mapping from pattern to the format to use (as passed to the format function).
            The first matching pattern is used.
        default_format: format used for all other metrics.
        exclude_keys: see the included_keys.
        include_keys: you can chose to exclude/include some metrics based on their name.
            If `include_keys` is non empty but `exclude_keys` is empty, then all keys
            are excluded by default and only those in `include_keys` are included (e.g. whitelist).
            The opposite (`exclude_keys` non empty, but `include_keys` empty), then
            this defines a blacklist. If both are provided, we first exclude then include back.
        include_formatted: if True (the default), implicitely include all metrics for which a format
            has been explicitely given in `formats`.
    """

    def __init__(
        self,
        formats: tp.Dict[str, str] = {},
        default_format: str = ".3f",
        exclude_keys: tp.Sequence[str] = [],
        include_keys: tp.Sequence[str] = [],
        include_formatted: bool = True,
    ):
        self.formats = dict(formats)
        self.default_format = default_format
        self.exclude_keys = list(exclude_keys)
        self.include_keys = list(include_keys)
        self.include_formatted = include_formatted

    def _is_excluded(self, key: str):
        for pattern in self.exclude_keys:
            if fnmatchcase(key, pattern):
                return True
        return False

    def _is_included(self, key: str):
        keys = self.include_keys
        if self.include_formatted:
            keys = keys + list(self.formats.keys())
        for pattern in keys:
            if fnmatchcase(key, pattern):
                return True
        return False

    def _get_format(self, key: str):
        for pattern, format_spec in self.formats.items():
            if fnmatchcase(key, pattern):
                return format_spec
        return self.default_format

    def get_relevant_metrics(self, metrics: dict) -> dict:
        def _keep_key(key):
            if self.exclude_keys:
                # exclude all keys in exclude_keys, then add back included ones.
                return not self._is_excluded(key) or self._is_included(key)
            elif self.include_keys:
                # Assume all keys are excluded except the one explicitely included.
                return self._is_included(key)
            else:
                return True
        return {k: v for k, v in metrics.items() if _keep_key(k)}

    def __call__(self, metrics: dict) -> dict:
        metrics = self.get_relevant_metrics(metrics)
        return {
            k: format(v, self._get_format(k)) for k, v in metrics.items()
        }
