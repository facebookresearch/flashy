# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utility class for automatically handling state of solver.
The object `StateManager()` can track stateful components of the solver.
Each component should follow the PyTorch idiomatic `state_dict()` and
`load_state_dict()` methods.

TODO: support strict vs. non strict loading. Support using only some
of the states, e.g. when fine tuning a checkpoint vs. continuing from
its own checkpoint.

The StateManager itself implements the state dict protocol.
"""

import typing as tp


StateDict = tp.Any  # we don't really care if those are really dicts or not


@tp.runtime_checkable
class StateDictSource(tp.Protocol):
    def state_dict(self) -> StateDict:
        ...

    def load_state_dict(self, state: StateDict):
        ...


class AttributeWrapper:
    """Turn any attribute into a `StateDictSource`."""
    def __init__(self, owner: tp.Any, name: str):
        self.owner = owner
        self.name = name

    def load_state_dict(self, state: StateDict):
        attr = getattr(self.owner, self.name)
        if isinstance(attr, StateDictSource):
            attr.load_state_dict(state)
        elif isinstance(attr, list):
            attr[:] = state
        elif isinstance(attr, dict):
            attr.clear()
            attr.update(state)
        else:
            setattr(self.owner, self.name, state)

    def state_dict(self):
        attr = getattr(self.owner, self.name)
        if isinstance(attr, StateDictSource):
            return attr.state_dict()
        else:
            return attr


class WriteOnlyWrapper(StateDictSource):
    def __init__(self, source: StateDictSource):
        self.source = source

    def load_state_dict(self, state):
        return

    def state_dict(self):
        return self.source.state_dict()


class StateManager(StateDictSource):
    def __init__(self):
        self.sources = {}

    def register(self, name: str, source: StateDictSource, write_only: bool = False):
        if name in self.sources:
            raise ValueError(f"{name} already present in sources.")
        if write_only:
            source = WriteOnlyWrapper(source)
        self.sources[name] = source

    def state_dict(self) -> StateDict:
        return {
            name: source.state_dict() for name, source in self.sources.items()
        }

    def load_state_dict(self, state: StateDict):
        for name, sub_state in state.items():
            self.sources[name].load_state_dict(sub_state)
