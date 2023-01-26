# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Most of the utilities here are derived from PyTorch Lightning
# Original LICENSE included hereafter
# Copyright 2018-2021 William Falcon
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Utilities for loggers."""
from argparse import Namespace
import typing as tp

import torch


def _fmt_prefix(prefix: tp.Union[str, tp.List[str]], delimiter: str = "/") -> str:
    """Format prefix(es) to a single prefix string."""
    if isinstance(prefix, str):
        return prefix
    else:
        return delimiter.join(prefix)


def _add_prefix(metrics: dict, prefix: tp.Union[str, tp.List[str]], delimiter: str = "/") -> dict:
    """Insert prefix before each key in a dict, separated by the delimiter.

    Args:
        metrics: Dictionary with metric names as keys and values
        prefix: Prefix to insert before each key
        delimiter: Separates prefix and original key name. Defaults to ``'/'``
    Returns:
        Dictionary with prefix and delimiter inserted before each key
    """
    if prefix is None:
        return metrics
    prefix = _fmt_prefix(prefix, delimiter)
    metrics = {f"{prefix}{delimiter}{k}": v for k, v in metrics.items()}
    return metrics


def _convert_params(params: tp.Union[dict, Namespace]) -> dict:
    """Ensure parameters are a dict or convert to dict if necessary.
    Args:
        params: Target to be converted to a dictionary
    Returns:
        params as a dictionary
    """
    # in case converting from namespace
    if isinstance(params, Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params


def _flatten_dict(params: tp.Dict[str, tp.Any], delimiter: str = ".") -> tp.Dict[str, tp.Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'.'``
    Returns:
        Flattened dict
    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(
        input_dict: tp.Any, prefixes: tp.Optional[tp.List[tp.Optional[str]]] = None
    ):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, tp.MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (tp.MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


def _sanitize_params(params: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    """Returns params with non-primitvies converted to strings for logging.
    >>> params = {"float": 0.3,
    ...           "int": 1,
    ...           "string": "abc",
    ...           "bool": True,
    ...           "list": [1, 2, 3],
    ...           "namespace": Namespace(foo=3),
    ...           "layer": torch.nn.BatchNorm1d}
    >>> import pprint
    >>> pprint.pprint(_sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
    {'bool': True,
        'float': 0.3,
        'int': 1,
        'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
        'list': '[1, 2, 3]',
        'namespace': 'Namespace(foo=3)',
        'string': 'abc'}
    """
    for k in params.keys():
        # convert relevant np scalars to python types first (instead of str)
        if type(params[k]) not in [bool, int, float, str, torch.Tensor]:
            params[k] = str(params[k])
    return params
