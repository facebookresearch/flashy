# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import os
import random
import multiprocessing as mp

import torch.distributed
from torch import nn

from flashy import distrib

WS = 8


def init(rank: int):
    os.environ['RANK'] = str(rank)
    torch.distributed.init_process_group(
        backend='gloo',
        init_method='env://')


def worker(rank: int):
    init(rank)

    x = torch.tensor([float(rank) + 1])
    distrib.average_tensors([x])
    assert x.item() == sum(range(1, WS + 1)) / WS, x.item()

    x = torch.tensor([float(rank) + 1])
    distrib.broadcast_tensors([x])
    assert x.item() == 1.

    y = torch.tensor([0.])
    try:
        if rank == 5:
            distrib.broadcast_tensors([x, y])
        else:
            distrib.broadcast_tensors([x])
    except RuntimeError:
        pass
    else:
        assert False, "Should have raised"

    mod = nn.Linear(1, 1, bias=False)
    mod.weight.data.zero_()
    x = torch.ones(1, 1)
    for eager in [False, True]:
        y = mod(x)
        gt = torch.tensor(float(rank)).view(-1, 1)
        loss = nn.functional.mse_loss(y, gt)
        if eager:
            with distrib.eager_sync_model(mod):
                loss.backward()
        else:
            loss.backward()
            distrib.sync_model(mod)
        grad = mod.weight.grad.data.clone()
        mod.weight.grad.data.zero_()
        y = mod(x.expand(WS, 1))
        gt = torch.arange(WS).float().view(-1, 1)
        loss = nn.functional.mse_loss(y, gt)
        loss.backward()
        grad_ref = mod.weight.grad.data
        assert torch.allclose(grad, grad_ref), (eager, grad.item(), grad_ref.item())
        mod.weight.grad.data.zero_()

    if distrib.rank() == 0:
        obj = defaultdict(int)
        obj['test'] = 42
        obj['youpi'] = 21
    else:
        obj = None
    received = distrib.broadcast_object(obj)
    assert isinstance(received, defaultdict)
    assert dict(received) == {'test': 42, 'youpi': 21}


def test_distrib():
    os.environ['WORLD_SIZE'] = str(WS)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(random.randrange(30000, 40000))
    ctx = mp.get_context('spawn')
    procs = []
    for rank in range(1, WS):
        procs.append(ctx.Process(target=worker, args=(rank,)))
        procs[-1].start()
    worker(0)
    for proc in procs:
        proc.join()
        assert proc.exitcode == 0
    del os.environ['WORLD_SIZE']
    del os.environ['MASTER_ADDR']
    del os.environ['MASTER_PORT']
    del os.environ['RANK']
