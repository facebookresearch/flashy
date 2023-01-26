# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Distrib utilities. When running from inside Dora, you can use the `init` function
imported from `dora.distrib`. Also works outside of Dora if distributed has been initialized
manually.
"""
from contextlib import contextmanager
from functools import partial, wraps
import pickle
import typing as tp

import torch
from torch import distributed
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset

from dora.distrib import rank, world_size, init  # noqa


def rank_zero_only(fn: tp.Callable) -> tp.Callable:
    """Function that can be used as a decorator to enable a
    function/method being called only on rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: tp.Any, **kwargs: tp.Any) -> tp.Optional[tp.Any]:
        if is_rank_zero():
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def is_rank_zero():
    return rank() == 0


def is_distributed():
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=distributed.ReduceOp.SUM):
    if is_distributed():
        return distributed.all_reduce(tensor, op)


def average_metrics(metrics: tp.Dict[str, float], count=1.):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))


def wrap(model):
    """Wrap a model in DDP if necessary. You can also choose to use `eager_sync_model`
    or `sync_model`.
    """
    if is_distributed():
        return DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device())
    else:
        return model


def _check_number_of_params(params: tp.List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    if not is_distributed() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(f"Mismatch in number of params: ours is {len(params)}, "
                           "at least one worker has a different one.")


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def average_tensors(tensors: tp.Iterable[torch.Tensor]):
    """All reduce averaging of the given tensor list.
    Note that non complex/floating point values are ignored.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.all_reduce(
            tensor.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
        handles.append((tensor, handle))
    for tensor, handle in handles:
        handle.wait()
        tensor.data /= world_size()


def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def broadcast_model(model: torch.nn.Module, src: int = 0):
    """Broadcast params and buffers from the given model to all workers."""
    broadcast_tensors(model.parameters(), src)
    broadcast_tensors(model.buffers(), src)


def sync_gradients(params: tp.Iterable[torch.Tensor]):
    """
    Average gradients from the given parameter list.
    This allows a simpler alternative to DistributedDataParallel.
    For a more complete alternative use `sync_model`, which also synchronizes
    buffers.

    See `eager_sync_gradients` for starting all reduce as soon as gradients
    are available during the backward.

    ..Warning:: This will only synchronize gradients, for full model synchronization
        including buffers, use `sync_model`.
    """
    grads = [param.grad for param in params if param.grad is not None]
    average_tensors(grads)


@contextmanager
def eager_sync_gradients(params: tp.Iterable[torch.Tensor]):
    """Similar to `sync_gradients`, except this is a context manager that will start syncing
    gradient as soon as they become available. This can be faster, but requires backward to be
    called no more than once!

    ..Warning:: This will only synchronize gradients, for full model synchronization
        including buffers, use `eager_sync_model`.
    """
    if not is_distributed():
        yield
        return
    params = list([p for p in params if p.requires_grad])
    _check_number_of_params(params)
    hooks = []
    handles = []
    waiting_params = set(params)

    def _callback(param, grad):
        if param not in waiting_params:
            raise RuntimeError(f"We got a gradient twice for parameter {param}.")
        handle = torch.distributed.all_reduce(grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
        handles.append((param, grad.data, handle))
        waiting_params.remove(param)

    for param in params:
        hooks.append(param.register_hook(partial(_callback, param)))

    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()
        _check_number_of_params(list(waiting_params))  # verify all workers have the same nb of remaining params.
        for param, grad, handle in handles:
            handle.wait()
            assert param.grad is not None
            torch.div(grad, world_size(), out=param.grad)


def sync_model(model: torch.nn.Module, sync_buffers: bool = True, average_buffers: bool = True):
    """
    Simpler alternative to DistributedDataParallel, that doesn't rely
    on any black magic. For simple models it can also be as fast.
    Just call this on your model after the backward is completed.

    Args:
        model: model to synchronize.
        sync_buffers: if True (the default), also synchronizes buffers.
        average_buffers: if True (the default), average buffers, instead
            broadcast from worker 0 (like DDP).
    """
    sync_gradients(model.parameters())
    if sync_buffers:
        if average_buffers:
            average_tensors(model.buffers())
        else:
            broadcast_tensors(model.buffers())


@contextmanager
def eager_sync_model(model: torch.nn.Module, sync_buffers: bool = True,
                     average_buffers: bool = True):
    """Same as `sync_model` but using `eager_sync_gradients`.
    """
    with eager_sync_gradients(model.parameters()):
        yield
    if sync_buffers:
        if average_buffers:
            average_tensors(model.buffers())
        else:
            broadcast_tensors(model.buffers())


def loader(dataset, *args, shuffle=False, klass=DataLoader, **kwargs):
    """
    Create a dataloader properly in case of distributed training.
    If a gradient is going to be computed you must set `shuffle=True`.
    """
    if not is_distributed():
        return klass(dataset, *args, shuffle=shuffle, **kwargs)

    if shuffle:
        # train means we will compute backward, we use DistributedSampler
        sampler = DistributedSampler(dataset)
        # We ignore shuffle, DistributedSampler already shuffles
        return klass(dataset, *args, **kwargs, sampler=sampler)
    else:
        # We make a manual shard, as DistributedSampler otherwise replicate some examples
        dataset = Subset(dataset, list(range(rank(), len(dataset), world_size())))
        return klass(dataset, *args, shuffle=shuffle, **kwargs)


def broadcast_object(obj: tp.Any = None, src: int = 0, device=None):
    """Share the given object (must be picklable) from the worker with rank `src`.
    """
    if not is_distributed():
        return obj
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = torch.empty(1, device=device, dtype=torch.long)
    if rank() == src:
        dump = bytearray(pickle.dumps(obj))
        # bytearray made a writable copy to avoid PyTorch warning later on.
        size[0] = len(dump)
    torch.distributed.broadcast(size, src=src)
    # size variable is now set to the length of pickled obj in all processes

    if rank() == src:
        buffer = torch.frombuffer(dump, dtype=torch.uint8).to(device=device)
    else:
        buffer = torch.empty(int(size[0].item()), device=device, dtype=torch.uint8)
    torch.distributed.broadcast(buffer, src=src)
    # buffer variable is now set to pickled obj in all processes
    if rank != src:
        obj = pickle.loads(buffer.cpu().numpy().tobytes())
    return obj


def barrier() -> None:
    """Barrier for all workers, when distributed is used.
    """
    if is_distributed():
        torch.distributed.barrier()
