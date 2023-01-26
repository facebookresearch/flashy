# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import subprocess as sp

from tests.dummy import train


def test_integ(tmpdir):
    environ = dict(os.environ)
    environ['_FLASHY_TMDIR'] = str(tmpdir)
    environ['DORA_PACKAGE'] = 'tests.dummy'
    kw = {'check': True, 'env': environ}

    sp.run(['dora', 'run', '--clear', 'stop_at=2'], **kw)
    train.main.dora.dir = str(tmpdir)
    xp = train.main.get_xp([])
    xp.link.load()
    assert len(xp.link.history) == 2
    old_history = list(xp.link.history)
    sp.run(['dora', 'run'], **kw)
    xp.link.load()
    assert len(xp.link.history) == 4
    assert xp.link.history[:2] == old_history

    sp.run(['dora', 'run', '--clear', '-d', '--ddp_workers=2'], **kw)
