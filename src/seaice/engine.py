"""Evaluation loop shared by training (val) and the final test run."""
from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from .metrics import FILL, STRATIFIED, ConfusionBank


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, eval_ds, device, tasks: dict[str, int], label_index: dict[str, int],
             batch_size: int = 4, workers: int = 0, criterion=None):
    """Score every pixel of eval_ds once. Returns (ConfusionBank, mean loss or None, seconds).

    Besides one bank per task, fills "SOD@90" / "FLOE@90": the same predictions
    scored only where the true SIC class is >= 9 (>= 90% concentration), where an
    ice-chart polygon label describes the pixel rather than the polygon average.
    """
    model.eval()
    dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    bank = ConfusionBank(len(eval_ds.store.scenes), {**tasks, **STRATIFIED})
    loss_sum, n_batches = 0.0, 0
    t0 = time.time()
    for x, y, scene, _, _ in dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = {t: out[t].argmax(1) for t in tasks}
        for t in tasks:
            bank.update(t, scene, y[:, label_index[t]], preds[t])
        sic = y[:, label_index["SIC"]]
        high = (sic >= 9) & (sic != FILL)
        for t in STRATIFIED:
            base = t.split("@")[0]
            yt = y[:, label_index[base]].clone()
            yt[~high] = FILL
            bank.update(t, scene, yt, preds[base])
        if criterion is not None:
            loss, _ = criterion(out, y)
            loss_sum += loss.item()
            n_batches += 1
    mean_loss = loss_sum / n_batches if criterion is not None and n_batches else None
    return bank, mean_loss, time.time() - t0
