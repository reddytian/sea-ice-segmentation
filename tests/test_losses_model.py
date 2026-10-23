import torch
import torch.nn.functional as F

from seaice.losses import MultiTaskCE, masked_ce
from seaice.model import TASKS, MultiTaskUNet


def test_pytorch_mean_ce_is_nan_when_every_target_is_ignored():
    """The failure masked_ce exists to prevent."""
    logits = torch.randn(2, 6, 4, 4)
    assert torch.isnan(F.cross_entropy(logits, torch.full((2, 4, 4), 255), ignore_index=255))


def test_masked_ce_matches_weighted_mean_and_is_zero_when_empty():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 8, 8, requires_grad=True)
    target = torch.randint(0, 6, (2, 8, 8))
    target[0, :4] = 255
    w = torch.rand(6) + 0.5
    assert torch.allclose(masked_ce(logits, target, w),
                          F.cross_entropy(logits, target, weight=w, ignore_index=255))
    empty = masked_ce(logits, torch.full_like(target, 255), w)
    empty.backward()
    assert empty.item() == 0 and torch.isfinite(logits.grad).all() and (logits.grad == 0).all()


def test_multitask_loss_with_one_task_unlabelled():
    crit = MultiTaskCE({"SIC": 0, "SOD": 1, "FLOE": 2})
    out = {t: torch.randn(2, n, 8, 8) for t, n in TASKS.items()}
    y = torch.stack([torch.randint(0, 11, (2, 8, 8)), torch.full((2, 8, 8), 255),
                     torch.randint(0, 7, (2, 8, 8))], 1)
    total, parts = crit(out, y)
    assert torch.isfinite(total) and parts["SOD"].item() == 0


def test_model_heads_and_size():
    m = MultiTaskUNet(in_ch=17, base=8)
    out = m(torch.randn(2, 17, 32, 32))
    assert {t: tuple(v.shape) for t, v in out.items()} == {
        "SIC": (2, 11, 32, 32), "SOD": (2, 6, 32, 32), "FLOE": (2, 7, 32, 32)}
    n = sum(p.numel() for p in MultiTaskUNet(in_ch=17, base=32).parameters())
    assert 7.7e6 < n < 7.8e6          # the configuration reported in the README
