"""Masked multi-task cross-entropy.

F.cross_entropy(..., ignore_index=255, reduction="mean") returns NaN when every
target in the batch is ignored (0/0). With SOD/FLOE unlabelled on ~35% of
SIC-labelled ice, a crop with SIC labels but no SOD labels is common, and one
NaN step corrupts the weights. masked_ce returns an exact zero (with a graph,
so backward() still works) when there is nothing to score.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

FILL = 255


def masked_ce(logits: torch.Tensor, target: torch.Tensor,
              weight: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted mean CE over non-fill pixels; 0 if there are none."""
    valid = target != FILL
    if not bool(valid.any()):
        return logits.sum() * 0.0
    total = F.cross_entropy(logits, target, weight=weight,
                            ignore_index=FILL, reduction="sum")
    denom = weight[target[valid]].sum() if weight is not None else valid.sum()
    return total / denom


class MultiTaskCE(torch.nn.Module):
    """Sum of per-task masked CE. `tasks` maps task name -> label channel index."""

    def __init__(self, tasks: dict[str, int], class_weights: dict[str, torch.Tensor] | None = None,
                 task_weights: dict[str, float] | None = None):
        super().__init__()
        self.tasks = tasks
        self.task_weights = task_weights or {t: 1.0 for t in tasks}
        for t in tasks:
            w = None if class_weights is None else class_weights[t]
            self.register_buffer(f"w_{t}", w)

    def forward(self, outputs: dict[str, torch.Tensor], y: torch.Tensor):
        parts = {t: masked_ce(outputs[t], y[:, k], getattr(self, f"w_{t}"))
                 for t, k in self.tasks.items()}
        total = sum(self.task_weights[t] * v for t, v in parts.items())
        # Detached tensors, not floats: float() forces a GPU sync per task per step.
        return total, {t: v.detach() for t, v in parts.items()}
