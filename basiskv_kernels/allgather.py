"""Prepared equal-width NCCL AllGather for feature-major TP coordinates.

Each rank writes [local_width, tokens] directly into its slice of one shared
arena. Gathering produces [world_size * local_width, tokens], ready for a single
decoder GEMM. Prepare each shape on the stream that will run it, before capture.
"""

from functools import lru_cache

import torch
import torch.distributed as dist

from basiskv_kernels.build import CSRC, load_cuda_extension


@lru_cache(maxsize=1)
def prepare_allgather_extension():
    return load_cuda_extension('basiskv_allgather', [CSRC / 'allgather.cpp'], nccl=True)


class AllGather:
    def __init__(self, implementation, *, rank, world_size, device):
        self._impl = implementation
        self.rank, self.world_size, self.device = rank, world_size, device
        self._workspace = None
        self._plans = {}

    @classmethod
    def from_distributed(cls, group=None, *, device=None):
        assert dist.is_initialized() and dist.get_backend(group) == 'nccl'
        device = torch.device('cuda', torch.cuda.current_device()) if device is None else torch.device(device)
        assert device.type == 'cuda'
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_device(index)
        extension = prepare_allgather_extension()
        rank, world_size = dist.get_rank(group), dist.get_world_size(group)
        size = extension.unique_id_size()
        identity = torch.zeros(size, dtype=torch.uint8, device=torch.device('cuda', index))
        if rank == 0:
            identity.copy_(torch.tensor(list(extension.unique_id()), dtype=torch.uint8))
        source = 0 if group is None else dist.get_global_rank(group, 0)
        dist.broadcast(identity, src=source, group=group)
        implementation = extension.Communicator(bytes(identity.cpu().tolist()), rank, world_size, index)
        return cls(implementation, rank=rank, world_size=world_size, device=torch.device('cuda', index))

    def configure_workspace(self, *, tokens, max_total_width, dtype):
        """Reserve one flat arena shared by sequential layers and token counts."""
        assert tokens > 0 and max_total_width > 0
        self._plans.clear()
        self._workspace = torch.empty(tokens * max_total_width, dtype=dtype, device=self.device)

    def prepare(self, local_width, *, tokens, dtype):
        assert local_width > 0 and tokens > 0
        assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.uint8)
        stream = torch.cuda.current_stream(self.device).cuda_stream
        key = (local_width, tokens, dtype, stream)
        if key not in self._plans:
            assert not torch.cuda.is_current_stream_capturing(), 'Prepare communication before CUDA Graph capture'
            total_width = local_width * self.world_size
            size = total_width * tokens
            if self._workspace is None:
                arena = torch.empty(total_width, tokens, device=self.device, dtype=dtype)
            else:
                assert self._workspace.dtype == dtype and size <= self._workspace.numel()
                arena = self._workspace[:size].view(total_width, tokens)
            self._plans[key] = self._impl.prepare(arena, local_width)
        return self._plans[key]

    def close(self):
        self._plans.clear()
        self._impl.close()
        self._workspace = None
