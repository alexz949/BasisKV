"""CPU smoke checks for routing-window construction; no model or dataset download.

    python -m unittest tests.test_routing_windows

Uses a reversible toy tokenizer and deterministic filler. Real tokenizer/C4
generation performs the same validation before writing its output bank.
"""

import collections
import hashlib
import random
import re
import sys
import types
import unittest
from unittest.mock import patch

from calibration.mix_windows import window_sources
from calibration.retrieval import FAMILIES, SEQUENCE_LENGTH, Haystack, build_window, validate


class ToyTokenizer:
    def __init__(self):
        self.vocab, self.pieces = {}, []

    def __call__(self, text, add_special_tokens=False):
        tokens = []
        for piece in re.findall(r'[A-Za-z]+|\d+|[^A-Za-z\d]', text):
            if piece not in self.vocab:
                self.vocab[piece] = len(self.pieces)
                self.pieces.append(piece)
            tokens.append(self.vocab[piece])
        return {'input_ids': tokens}

    def decode(self, tokens):
        return ''.join(self.pieces[i] for i in tokens)


class ToyHaystack:
    def __init__(self, tokenizer):
        self.tokens = tokenizer('Ordinary filler about trees, rivers and clouds. ' * 100)['input_ids']
        self.index = 0
        self.rng = random.Random(20260923)

    def excerpt(self):
        self.index += 1
        return f'document-{self.index}', self.index, self.tokens[:self.rng.randint(256, 1500)]


class RoutingWindowsTest(unittest.TestCase):
    def test_haystack_excludes_c4_documents_and_duplicates(self):
        excluded = 'Held out calibration document. ' * 100
        first, second = 'First filler document. ' * 100, 'Second filler document. ' * 100
        rows = [dict(text=t) for t in (excluded, 'Too short.', first, first, second)]

        class Stream:
            def shuffle(self, **kwargs):
                return self

            def __iter__(self):
                return iter(rows)

        digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
        dataset = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: Stream())
        with patch.dict(sys.modules, datasets=dataset):
            haystack = Haystack(ToyTokenizer(), 20260923, {digest(excluded)})
            self.assertEqual(haystack.excerpt()[0], digest(first))
            self.assertEqual(haystack.excerpt()[0], digest(second))

    def test_full_mix_and_supports(self):
        tokenizer = ToyTokenizer()
        haystack = ToyHaystack(tokenizer)
        windows, rows = [], []
        for index, family in enumerate(FAMILIES):
            window, row = build_window(index, family, 20260922000 + index, tokenizer, haystack)
            windows.append(window)
            rows.append(row)
            self.assertEqual(len(window), 131072)
            self.assertEqual(len(row['queries']), row['interactions'])
            self.assertGreaterEqual(row['interactions'], 48)
            self.assertLessEqual(row['interactions'], 96)
            for record in row['records']:
                text = tokenizer.decode(window[record['token_start']:record['token_end']])
                self.assertEqual(text, '\n' + record['text'] + '\n')
            for query in row['queries']:
                self.assertGreaterEqual(query['query_token_position'], row['tail_start'])
                self.assertTrue(all(p < row['tail_start'] for p in query['support_token_positions']))
                self.assertEqual(query['distances'], [query['query_token_position'] - p for p in query['support_token_positions']])
        self.assertEqual(collections.Counter(row['task_family'] for row in rows),
                         {'multikey': 8, 'multivalue': 4, 'tracking': 2, 'aggregation': 2})
        report = validate(windows, rows, tokenizer)
        for key in ('length_ok', 'answers_ok', 'leakage_ok', 'collision_ok'):
            self.assertTrue(report[key], key)

    def test_seed_reproducibility(self):
        def generate():
            tokenizer = ToyTokenizer()
            window, row = build_window(0, 'multikey', 20260922000, tokenizer, ToyHaystack(tokenizer))
            return tokenizer.decode(window), row
        self.assertEqual(generate(), generate())

    def test_mix_preserves_heldout_windows(self):
        # Non-contiguous source IDs verify that assembly follows the manifest.
        c4 = dict(shape=[48, SEQUENCE_LENGTH], fit_ids=list(range(0, 48, 3)) + list(range(1, 48, 3)),
                  validation_ids=list(range(2, 48, 3)))
        synthetic = dict(shape=[16, SEQUENCE_LENGTH], families=FAMILIES)
        sources = window_sources(c4, synthetic)
        self.assertEqual(sources[:16], [('c4', i) for i in c4['fit_ids'][:16]])
        self.assertEqual(sources[16:32], [('synthetic', i) for i in range(16)])
        self.assertEqual(sources[32:], [('c4', i) for i in c4['validation_ids']])
        self.assertTrue(set(sources[:32]).isdisjoint(sources[32:]))


if __name__ == '__main__':
    unittest.main()
