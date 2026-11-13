"""Page4 system routing checks: python -m unittest tests.test_page4_router -v.

CPU checks cover exact group pruning; CUDA checks compile the production kernels
and compare selection, page scores, Key-slot refresh, and attention with references.
"""

import importlib.util
import math
import unittest

import numpy as np

from kernels.routing_config import PAGE_SIZE, RECENT_TOKENS, ROUTED_PAGES, SUPPORT


class GroupSelectionTest(unittest.TestCase):
    def test_group_pruning_preserves_top_pages(self):
        rng = np.random.default_rng(42)
        for pages in (513, 1025, 32769, 65536):
            for tied in (False, True):
                scores = np.zeros((4, pages)) if tied else rng.normal(size=(4, pages))
                probabilities = np.exp(scores - scores.max(-1, keepdims=True))
                probabilities /= probabilities.sum(-1, keepdims=True)
                values = probabilities.max(0)
                order = np.lexsort((np.arange(pages), -values))[:ROUTED_PAGES]
                grouped = np.pad(values, (0, (-pages) % 8), constant_values=-np.inf).reshape(-1, 8).max(-1)
                threshold = np.partition(grouped, -min(ROUTED_PAGES, len(grouped)))[-min(ROUTED_PAGES, len(grouped))]
                candidates = np.flatnonzero(np.repeat(grouped >= threshold, 8)[:pages])
                selected = candidates[np.lexsort((candidates, -values[candidates]))[:ROUTED_PAGES]]
                np.testing.assert_array_equal(np.sort(selected), np.sort(order))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CUDA checks require PyTorch')
class CudaRouterTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            self.skipTest('requires an sm_80+ CUDA GPU')
        from kernels.decode import load_router
        self.torch = torch
        self.router = load_router(128)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False

    def buffers(self, groups=1):
        t = self.torch
        return (t.empty(1, groups, ROUTED_PAGES, device='cuda', dtype=t.long),
                t.empty(1, groups, SUPPORT, device='cuda', dtype=t.long),
                t.empty(1, groups, self.router.selection_workspace_size(), device='cuda'))

    def test_selection_large_contexts_and_partial_pages(self):
        t = self.torch
        for historical in (4097, 131089, 262144):
            pages = math.ceil(historical / PAGE_SIZE)
            selected, support, scratch = self.buffers(2)
            for tied in (False, True):
                scores = t.zeros(1, 2, 4, pages + 3, device='cuda') if tied else t.randn(1, 2, 4, pages + 3, device='cuda')
                scores = scores[..., :pages]  # exercise the benchmark's strided output view
                scores[..., 0] = -10000  # page zero must not be pinned
                mass = scores.softmax(-1).amax(2)
                expected_pages = mass.argsort(-1, descending=True, stable=True)[..., :ROUTED_PAGES].sort(-1).values
                self.router.select_pages(scores, selected, support, scratch, historical, historical + RECENT_TOKENS)
                t.testing.assert_close(selected, expected_pages, rtol=0, atol=0)
                expected = (expected_pages[..., None] * PAGE_SIZE + t.arange(PAGE_SIZE, device='cuda')).flatten(-2)
                expected.masked_fill_(expected >= historical, -1)
                recent = t.arange(historical, historical + RECENT_TOKENS, device='cuda').expand(1, 2, -1)
                t.testing.assert_close(support, t.cat((expected, recent), -1), rtol=0, atol=0)

    def test_page_scores(self):
        t = self.torch
        for tokens in (5, 129, 131073):
            def random(*shape):
                return (t.randn(*shape, device='cuda') * 0.2).bfloat16()
            query, base, residual = random(1, 4, 1, 128), random(1, 1, tokens, 16), random(1, 1, tokens, 16)
            right, bias, projector = random(1, 16, 128), random(1, 128), random(4, 128, 16)
            angles = t.randn(tokens, 64, device='cuda')
            cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
            code = t.empty(1, 1, 4, 16, device='cuda', dtype=t.bfloat16)
            output = t.empty(1, 1, 4, math.ceil(tokens / PAGE_SIZE), device='cuda')
            self.router.page_lse(query, base, residual, right, bias, projector, cos, sin, code, output, 128**-0.5)
            predicted = (base @ right) + bias[:, None, :]
            low, high = predicted[..., :64], predicted[..., 64:]
            rotated = t.cat((low * cos - high * sin, high * cos + low * sin), -1)
            qcode = (query[:, :, 0, None, :] @ projector).squeeze(-2)
            expected = ((query[:, :, 0] @ rotated[:, 0].transpose(-1, -2)) +
                        (qcode @ residual[:, 0].transpose(-1, -2))) * (128**-0.5)
            expected = t.nn.functional.pad(expected.float(), (0, (-tokens) % PAGE_SIZE), value=-t.inf)
            expected = expected.reshape(1, 1, 4, -1, PAGE_SIZE).logsumexp(-1)
            t.testing.assert_close(output, expected, rtol=0.01, atol=0.01)

    def test_host_slots_and_2112_token_attention(self):
        t = self.torch
        from kernels.sparse_attention import sparse_attention
        capacity = 131200
        host = self.router.host_key_buffer(1, 1, capacity)
        host.copy_(t.randn(1, 1, capacity, 128, dtype=t.bfloat16))
        pointer = self.router.device_pointer(host)
        selected, support, scratch = self.buffers()
        cache = t.empty(1, 1, SUPPORT, 128, device='cuda', dtype=t.bfloat16)
        resident = t.full_like(support, -1)
        lookup = t.full((1, 1, capacity), -1, device='cuda', dtype=t.int32)
        slots = t.empty_like(support)
        missing = t.empty_like(support, dtype=t.int32)
        counts = t.empty(1, 1, 2, device='cuda', dtype=t.int32)
        for historical in (131073, 131074):
            scores = t.randn(1, 1, 4, math.ceil(historical / PAGE_SIZE), device='cuda')
            self.router.select_pack_refresh(scores, selected, support, pointer, cache,
                resident, lookup, slots, missing, counts, scratch, historical, historical + RECENT_TOKENS)
            t.cuda.synchronize()
            valid = support[0, 0] >= 0
            actual_keys = cache[0, 0, slots[0, 0, valid]].cpu()
            expected_keys = host[0, 0, support[0, 0, valid].cpu()]
            t.testing.assert_close(actual_keys, expected_keys, rtol=0, atol=0)
            self.assertEqual(int(counts[0, 0, 1]), int(valid.sum()))
        query = t.randn(1, 4, 1, 128, device='cuda', dtype=t.bfloat16)
        value = t.randn(1, 1, capacity, 96, device='cuda', dtype=t.bfloat16)
        actual = sparse_attention(query, cache, value, support, key_slots=slots, scale=128**-0.5)
        keys = cache[0, 0, slots[0, 0, valid]].float()
        values = value[0, 0, support[0, 0, valid]].float()
        expected = ((query[0, :, 0].float() @ keys.T) * (128**-0.5)).softmax(-1) @ values
        t.testing.assert_close(actual[0, :, 0].float(), expected, rtol=0.02, atol=0.002)

    def test_dense_page4_attention(self):
        t = self.torch
        from kernels.decode import load_paged_attention
        attention = load_paged_attention(96, 4)
        length = 1003
        query = t.randn(1, 4, 1, 128, device='cuda', dtype=t.bfloat16)
        keys = t.randn(1, 1, length, 128, device='cuda', dtype=t.bfloat16)
        values = t.randn(1, 1, length, 96, device='cuda', dtype=t.bfloat16)
        pages = t.arange(math.ceil(length / PAGE_SIZE), device='cuda').view(1, 1, -1)
        workspace = t.empty(4, 16, 98, device='cuda')
        output = t.empty(1, 4, 1, 96, device='cuda', dtype=t.bfloat16)
        attention.attention(keys, query, values, pages, workspace, output, length, 128**-0.5, 16)
        expected = ((query[0, :, 0].float() @ keys[0, 0].float().T) * (128**-0.5)).softmax(-1) @ values[0, 0].float()
        t.testing.assert_close(output[0, :, 0].float(), expected, rtol=0.02, atol=0.002)


if __name__ == '__main__':
    unittest.main()
