import ast
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


class TestEvalNanDiagnostic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the reporting helpers without installing the opt-in Trainer patch.
        path = Path(__file__).resolve().parents[2] / 'Nepham/CCLLM/diagnose_eval_nan.py'
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'tensor_summary', 'batch_report'}]
        namespace = {'torch': torch, 'F': F}
        exec(compile(tree, str(path), 'exec'), namespace)
        cls.batch_report = staticmethod(namespace['batch_report'])

    def test_empty_shifted_targets_are_identified(self):
        report = self.batch_report(torch.tensor(float('nan')), torch.zeros(1, 1, 3), {'labels': torch.tensor([[1]])})
        self.assertEqual(report['valid_targets_per_row'], [0])
        self.assertEqual(report['logits']['nonfinite'], 0)
        self.assertIsNone(report['reference_ce'])

    def test_finite_reference_loss_with_nonfinite_original_loss(self):
        logits = torch.zeros(1, 4, 3)
        labels = torch.tensor([[-100, 1, -100, 2]])
        report = self.batch_report(torch.tensor(float('nan')), logits, {'labels': labels})
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, 3), labels[:, 1:].reshape(-1))
        self.assertEqual(report['valid_targets_per_row'], [2])
        self.assertAlmostEqual(float(report['reference_ce']), expected.item())

    def test_nonfinite_logits_are_identified(self):
        logits = torch.zeros(1, 2, 3)
        logits[0, 0, 0] = float('nan')
        report = self.batch_report(torch.tensor(float('nan')), logits, {'labels': torch.tensor([[-100, 1]])})
        self.assertEqual(report['logits']['nonfinite'], 1)
        self.assertEqual(report['reference_ce'], 'nan')

    def test_nonfinite_ignored_logits_do_not_contaminate_reference(self):
        logits = torch.zeros(1, 3, 3)
        logits[0, 0, 0] = float('nan')
        report = self.batch_report(torch.tensor(float('nan')), logits, {'labels': torch.tensor([[-100, -100, 1]])})
        self.assertEqual(report['logits']['nonfinite'], 1)
        self.assertTrue(torch.isfinite(torch.tensor(float(report['reference_ce']))))

    def test_reduced_logits_and_invalid_targets_are_not_scored(self):
        for inputs in [{'labels': torch.tensor([[-100, 1]]), 'logits_to_keep': torch.tensor([True, False])},
                       {'labels': torch.tensor([[-100, 5]])}]:
            with self.subTest(inputs=inputs):
                report = self.batch_report(torch.tensor(float('nan')), torch.zeros(1, 2, 3), inputs)
                self.assertIsNone(report['reference_ce'])
                self.assertIn('reference_ce_skipped', report)


if __name__ == '__main__':
    unittest.main()
