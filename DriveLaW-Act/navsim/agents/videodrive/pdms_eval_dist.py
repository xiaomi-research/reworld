"""Distributed helpers for navtest PDMS evaluation."""

from __future__ import annotations

import pickle
from typing import Any, Dict, List

import torch
import torch.distributed as dist


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if dist_ready() else 0


def world_size() -> int:
    return dist.get_world_size() if dist_ready() else 1


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size: int):
        self._size = int(size)
        assert size > 0
        self._rank = rank()
        self._world_size = world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size: int, world_size: int, rank: int):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    if not dist_ready():
        return obj
    if rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor), device=device, dtype=torch.long)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0, device=device, dtype=torch.long)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(int(size_tensor.item())).to(device)
        dist.broadcast(tensor, src=src)
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    return obj


def gather_pickled_rows(local_rows: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    if not dist_ready():
        return local_rows

    payload = pickle.dumps(local_rows)
    local_tensor = torch.ByteTensor(list(payload)).to(device)
    local_size = torch.tensor([local_tensor.numel()], device=device, dtype=torch.long)
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    sizes = [int(s.item()) for s in size_list]
    max_size = max(sizes)

    if local_tensor.numel() < max_size:
        padded = torch.cat(
            [local_tensor, torch.zeros(max_size - local_tensor.numel(), dtype=torch.uint8, device=device)]
        )
    else:
        padded = local_tensor

    gathered = [torch.empty(max_size, dtype=torch.uint8, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)

    if rank() != 0:
        return []

    merged: List[Dict[str, Any]] = []
    for tensor, size in zip(gathered, sizes):
        data = tensor[:size].cpu().numpy().tobytes()
        merged.extend(pickle.loads(data))
    return merged
