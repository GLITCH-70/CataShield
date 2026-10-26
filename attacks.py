"""
attacks.py
==========
White-box, untargeted, L-infinity adversarial attacks on semantic segmentation (spec section 25).

Threat model
  - The attacker knows the model and its weights (white-box) and the ground-truth mask, and wants
    to make as many pixels as possible wrong (untargeted).
  - The perturbation delta lives in [0, 1] RGB pixel space, BEFORE ImageNet normalisation (the
    model normalises internally): ||delta||_inf <= eps and x + delta in [0, 1].
  - eps is quoted in 8-bit units, e.g. eps = 4/255.

Attacks
  - FGSM   (Goodfellow et al., ICLR 2015): one step of size eps along sign(grad of pixel CE).
  - PGD    (Madry et al., ICLR 2018): uniform random start in the eps-ball, T steps of size alpha
           along sign(grad of pixel CE), projection after every step.
  - SegPGD (Gu et al., ECCV 2022): PGD on a reweighted loss. At step t = 1..T
               L_t = (1 - lambda_t) * sum_{correct pixels} CE + lambda_t * sum_{wrong pixels} CE,
               lambda_t = (t - 1) / (2T),
           so early steps push still-correct pixels over the boundary instead of spending gradient
           on pixels that are already misclassified, which plain pixel-averaged CE does.
  - Random noise control: delta = eps * random sign per pixel/channel. Same L-inf budget as the
           attacks with no gradient information; separates adversarial direction from mere noise.

Implementation notes
  - The pixel losses are summed rather than averaged. Only sign(grad) is used, so the scaling does
    not change the attack, but it keeps per-pixel gradients well away from float underflow.
  - Attacks run in full fp32 with the model in eval mode (BatchNorm uses running statistics).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from cataract_dataset import IGNORE_INDEX


def _project(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    """Project onto {x' : ||x' - x||_inf <= eps, x' in [0, 1]}."""
    return torch.min(torch.max(x_adv, x - eps), x + eps).clamp_(0.0, 1.0)


def _input_grad(loss: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    (grad,) = torch.autograd.grad(loss, x_adv)
    return grad


def fgsm(model, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y, ignore_index=IGNORE_INDEX, reduction="sum")
    grad = _input_grad(loss, x_adv)
    return _project(x + eps * grad.sign(), x, eps).detach()


def pgd(
    model, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 20,
    alpha: Optional[float] = None, random_start: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    alpha = eps / 4 if alpha is None else alpha
    x_adv = x.clone()
    if random_start:
        noise = torch.rand(x.shape, generator=generator, device=x.device) * 2 - 1
        x_adv = _project(x + eps * noise, x, eps)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y, ignore_index=IGNORE_INDEX, reduction="sum")
        grad = _input_grad(loss, x_adv)
        x_adv = _project(x_adv.detach() + alpha * grad.sign(), x, eps)
    return x_adv.detach()


def segpgd_lambda(t: int, steps: int) -> float:
    """lambda_t = (t - 1) / (2T) for t = 1..T (Gu et al., 2022, Eq. 4)."""
    return (t - 1) / (2 * steps)


def segpgd(
    model, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 20,
    alpha: Optional[float] = None, random_start: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    alpha = eps / 4 if alpha is None else alpha
    valid = y != IGNORE_INDEX
    x_adv = x.clone()
    if random_start:
        noise = torch.rand(x.shape, generator=generator, device=x.device) * 2 - 1
        x_adv = _project(x + eps * noise, x, eps)
    for t in range(1, steps + 1):
        lam = segpgd_lambda(t, steps)
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        ce = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX, reduction="none")  # [B, H, W]
        correct = (logits.argmax(1) == y) & valid
        wrong = (~correct) & valid
        loss = (1 - lam) * ce[correct].sum() + lam * ce[wrong].sum()
        grad = _input_grad(loss, x_adv)
        x_adv = _project(x_adv.detach() + alpha * grad.sign(), x, eps)
    return x_adv.detach()


def random_sign_noise(
    x: torch.Tensor, eps: float, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    signs = torch.randint(0, 2, x.shape, generator=generator, device=x.device).float() * 2 - 1
    return _project(x + eps * signs, x, eps)


def run_attack(
    name: str, model, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 20,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if name == "clean" or eps == 0:
        return x
    if name == "noise":
        return random_sign_noise(x, eps, generator)
    if name == "fgsm":
        return fgsm(model, x, y, eps)
    if name == "pgd":
        return pgd(model, x, y, eps, steps, generator=generator)
    if name == "segpgd":
        return segpgd(model, x, y, eps, steps, generator=generator)
    raise ValueError(f"unknown attack '{name}'")


def check_perturbation(x: torch.Tensor, x_adv: torch.Tensor, eps: float, tol: float = 1e-6) -> float:
    """Assert the L-inf budget and valid pixel range; returns the measured ||delta||_inf."""
    linf = float((x_adv - x).abs().max())
    assert linf <= eps + tol, f"L-inf budget violated: {linf * 255:.4f}/255 > {eps * 255:.4f}/255"
    assert float(x_adv.min()) >= 0.0 and float(x_adv.max()) <= 1.0, "adversarial image left [0, 1]"
    return linf


if __name__ == "__main__":
    # Self-test on a small input. Uses the trained checkpoint if it exists, otherwise an
    # ImageNet-initialised model, and verifies budgets plus that each attack raises the loss.
    import os

    from model_builder import build_cataract_model, load_trained_model

    torch.manual_seed(0)
    device = "cpu"
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "baseline", "best_model.pth")
    if os.path.isfile(ckpt):
        model = load_trained_model(ckpt, device)
    else:
        model = build_cataract_model(pretrained=True, device=device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x).argmax(1)  # attack the model's own clean prediction
    eps = 4 / 255
    with torch.no_grad():
        base = F.cross_entropy(model(x), y).item()
    print(f"clean CE vs own prediction: {base:.4f}")
    gen = torch.Generator().manual_seed(1)
    for name in ("noise", "fgsm", "pgd", "segpgd"):
        x_adv = run_attack(name, model, x, y, eps, steps=5, generator=gen)
        linf = check_perturbation(x, x_adv, eps)
        with torch.no_grad():
            logits = model(x_adv)
            ce = F.cross_entropy(logits, y).item()
            flipped = (logits.argmax(1) != y).float().mean().item()
        print(f"{name:7s} ||d||_inf = {linf * 255:.3f}/255   CE {ce:.4f}   pixels flipped {100 * flipped:5.1f}%")
        if name != "noise":
            assert ce > base, f"{name} did not increase the loss"
    assert segpgd_lambda(1, 20) == 0.0 and abs(segpgd_lambda(20, 20) - 19 / 40) < 1e-12
    print("[SUCCESS] budgets respected, attacks increase the loss, SegPGD schedule correct")
