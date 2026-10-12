"""Load with --external_plugins to diagnose the first evaluation, then stop before training."""
import inspect
import json
import os
from functools import wraps
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

import swift
from swift.dataset.utils import EncodePreprocessor
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer


def tensor_summary(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None
    flat = tensor.detach().reshape(-1)
    nonfinite = sum(int((~torch.isfinite(chunk)).sum().item()) for chunk in flat.split(1048576))
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'nonfinite': nonfinite}


def batch_report(loss, logits, inputs):
    labels = inputs.get('labels')
    report = {
        'loss': tensor_summary(loss),
        'loss_value': str(loss.detach().cpu().tolist()),
        'logits': tensor_summary(logits),
        'labels': tensor_summary(labels),
        'loss_scale': tensor_summary(inputs.get('loss_scale')),
        'reference_ce': None,
    }
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        report['reference_ce_skipped'] = 'Missing two-dimensional labels'
        return report
    targets = labels[:, 1:]
    report['valid_targets_per_row'] = (targets != -100).sum(dim=-1).tolist()
    # Reduced logits may use a different alignment; do not guess their target mapping.
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[:2] != labels.shape
            or inputs.get('logits_to_keep') is not None):
        report['reference_ce_skipped'] = 'Logits missing, reduced or not aligned with labels'
        return report
    valid_targets = targets[targets != -100]
    if valid_targets.numel() == 0:
        report['reference_ce_skipped'] = 'No valid shifted targets'
        return report
    report['target_id_range'] = [int(valid_targets.min().item()), int(valid_targets.max().item())]
    if report['target_id_range'][0] < 0 or report['target_id_range'][1] >= logits.shape[-1]:
        report['reference_ce_skipped'] = 'Target outside logits vocabulary'
        return report
    total_nll = 0.
    for row in range(labels.shape[0]):
        for start in range(0, targets.shape[1], 128):
            target = targets[row, start:start + 128].to(logits.device)
            mask = target != -100
            if not mask.any().item():
                continue
            scores = logits[row, start:start + target.numel()][mask].float()
            total_nll += F.cross_entropy(scores, target[mask], reduction='sum').item()
    report['reference_ce'] = str(total_nll / valid_targets.numel())
    return report


def save_report(trainer, model, inputs, loss, outputs):
    rank = os.environ.get('RANK', '0')
    destination = Path(trainer.args.output_dir) / f'nan_diagnostic_rank{rank}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    logits = outputs.get('logits') if isinstance(outputs, dict) else getattr(outputs, 'logits', None)
    report = batch_report(loss, logits, inputs)
    model_loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
    report['output_loss'] = tensor_summary(model_loss)
    base_model = trainer.accelerator.unwrap_model(model)
    report.update({
        'rank': rank,
        'global_step': trainer.state.global_step,
        'swift_version': swift.__version__,
        'torch_version': torch.__version__,
        'transformers_version': transformers.__version__,
        'preprocessor_file': inspect.getsourcefile(EncodePreprocessor),
        'preprocessor_source': inspect.getsource(EncodePreprocessor.preprocess),
        'model_class': type(base_model).__name__,
        'model_file': inspect.getsourcefile(type(base_model)),
        'loss_function': str(getattr(base_model, 'loss_function', None)),
        'template': type(trainer.template).__name__,
        'padding_free': trainer.template.padding_free,
        'logits_to_keep': str(inputs.get('logits_to_keep')),
        'input_keys': sorted(inputs),
    })
    # Keep only tensor inputs; no model weights, logits or executable callback objects.
    batch = {key: value.detach().cpu() for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    torch.save(batch, destination.with_suffix('.pt'))
    destination.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[nan-diagnostic] {destination}.json\n{json.dumps(report, ensure_ascii=False)}', flush=True)
    return destination


def install():
    if getattr(Seq2SeqTrainer.compute_loss, '_nan_diagnostic', False):
        return
    original_compute_loss = Seq2SeqTrainer.compute_loss
    original_evaluate = Seq2SeqTrainer.evaluate

    @wraps(original_compute_loss)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.model.training:
            return original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
        # compute_loss pops labels/loss_scale from inputs on some loss paths.
        snapshot = dict(inputs)
        if isinstance(snapshot.get('labels'), torch.Tensor):
            snapshot['labels'] = snapshot['labels'].detach().clone()
        loss, outputs = original_compute_loss(self, model, inputs, True, num_items_in_batch)
        if not torch.isfinite(loss).all().item():
            with torch.no_grad():
                destination = save_report(self, model, snapshot, loss, outputs)
            raise RuntimeError(f'Nonfinite evaluation loss; diagnostic saved to {destination}.json')
        return (loss, outputs) if return_outputs else loss

    @wraps(original_evaluate)
    def evaluate(self, *args, **kwargs):
        metrics = original_evaluate(self, *args, **kwargs)
        print(f'[nan-diagnostic] Evaluation completed: {metrics}', flush=True)
        raise RuntimeError('Diagnostic evaluation finished; stopping before training. '
                           'Remove the diagnostic plugin to train.')

    compute_loss._nan_diagnostic = True
    Seq2SeqTrainer.compute_loss = compute_loss
    Seq2SeqTrainer.evaluate = evaluate
    print(f'[nan-diagnostic] enabled; swift={swift.__file__}; preprocessor={inspect.getsourcefile(EncodePreprocessor)}',
          flush=True)


install()
