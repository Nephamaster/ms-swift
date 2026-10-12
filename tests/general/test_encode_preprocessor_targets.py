# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from swift.dataset.preprocessor import RowPreprocessor
from swift.dataset.utils import AddLengthPreprocessor, EncodePreprocessor
from swift.template import Template


class TestEncodePreprocessorTargets(unittest.TestCase):

    def _preprocessor(self, encoded, task_type='causal_lm', strategy='split'):
        template = SimpleNamespace(
            task_type=task_type, truncation_strategy=strategy, encode=Mock(return_value=encoded))
        return EncodePreprocessor(template)

    def test_split_boundaries(self):
        for length, expected in [(2048, [2048]), (2049, [2048]), (2050, [2048, 2]), (4097, [2048, 2048])]:
            with self.subTest(length=length):
                encoded = {'input_ids': [7] * length, 'labels': [7] * length}
                template = SimpleNamespace(
                    mode='train',
                    task_type='causal_lm',
                    max_length=2048,
                    truncation_strategy='split',
                    _preprocess_inputs=lambda inputs: None,
                    _encode=lambda inputs: encoded,
                    _get_length=Template._get_length)
                chunks = Template._encode_truncated(template, None)
                result = self._preprocessor(chunks).preprocess({})
                self.assertEqual([len(item['input_ids']) for item in result], expected)
                self.assertTrue(all(any(label != -100 for label in item['labels'][1:]) for item in result))

    def test_unsplit_rows_without_shifted_targets_are_removed(self):
        for labels in [[], [-100], [7], [-100, -100], [7, -100, -100]]:
            with self.subTest(labels=labels):
                self.assertEqual(self._preprocessor({'labels': labels}).preprocess({}), [])

    def test_valid_rows_keep_metadata_and_order(self):
        first = {'input_ids': [3, 7], 'labels': [-100, 7], 'lengths': [2], 'loss_scale': [0., 1.]}
        last = {'input_ids': [3, 8, 9], 'labels': [-100, -100, 9], 'lengths': [3]}
        result = self._preprocessor([first, {'labels': [7, -100]}, last]).preprocess({})
        self.assertEqual(result, [first, last])
        self.assertIs(result[0], first)
        self.assertIs(result[1], last)
        self.assertEqual(self._preprocessor(first).preprocess({}), [first])

    def test_all_invalid_chunks_are_removed(self):
        self.assertEqual(self._preprocessor([{'labels': [-100]}, {'labels': [7, -100]}]).preprocess({}), [])

    def test_other_tasks_and_strategies_are_unchanged(self):
        for task_type, strategy in [('seq_cls', 'split'), ('embedding', 'split'), ('causal_lm', 'right'),
                                    ('causal_lm', 'left'), ('causal_lm', 'raise')]:
            with self.subTest(task_type=task_type, strategy=strategy):
                encoded = {'lengths': [1]}
                preprocessor = self._preprocessor(encoded, task_type=task_type, strategy=strategy)
                self.assertIs(preprocessor.preprocess({}), encoded)

    def test_add_length_preprocessor_keeps_unsplit_contract(self):
        template = self._preprocessor({'lengths': [2]}, strategy='right').template
        row = {'messages': [{'role': 'assistant', 'content': 'hello'}]}
        self.assertIs(AddLengthPreprocessor(template).preprocess(row), row)
        self.assertEqual(row['lengths'], [2])

    def test_encoded_cache_does_not_reuse_legacy_filename(self):
        preprocessor = self._preprocessor({'labels': [-100, 7]})
        self.assertNotEqual(preprocessor._map_cache_file('same-input'),
                            RowPreprocessor()._map_cache_file('same-input'))


if __name__ == '__main__':
    unittest.main()
