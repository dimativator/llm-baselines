"""Frozen SVD-LLM bases with trainable singular-value vectors."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from compression.svd_llm import stable_cholesky_whitening
from models.compress import effective_rank


class SigmaLinear(nn.Module):
    def __init__(self, left: torch.Tensor, sigma: torch.Tensor,
                 right: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.register_buffer("left", left.float())
        self.register_buffer("right", right.float())
        self.register_buffer("bias", bias.detach().clone() if bias is not None else None)
        self.sigma = nn.Parameter(sigma.float())

    @property
    def weight(self) -> torch.Tensor:
        return (self.left * self.sigma) @ self.right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.right) * self.sigma.to(x.dtype), self.left, self.bias)


@torch.no_grad()
def factorize(model: nn.Module, covariances: dict[str, torch.Tensor]) -> dict:
    """Use exact rounded weight erank, without the optional residual rank increase."""
    model.requires_grad_(False)
    records = {}
    for name, module in list(model.named_modules()):
        if not name.startswith("transformer.h.") or not isinstance(module, nn.Linear):
            continue
        weight = module.weight.detach().float()
        erank = effective_rank(weight)
        rank = max(1, min(round(erank), min(weight.shape)))
        covariance = covariances[name + ".weight"].to(weight.device)
        whitening, inverse = stable_cholesky_whitening(covariance)
        left, sigma, right = torch.linalg.svd(weight.double() @ whitening, full_matrices=False)
        # The right basis is de-whitened, so it need not be orthogonal in weight space.
        replacement = SigmaLinear(left[:, :rank], sigma[:rank], right[:rank] @ inverse, module.bias)
        replacement.to(weight.device)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child_name, replacement)
        records[name] = dict(original_effective_rank=erank, retained_rank=rank,
                             shape=list(weight.shape))
    if not records:
        raise RuntimeError("No transformer matrices factorized")
    assert all(name.endswith(".sigma") for name, p in model.named_parameters() if p.requires_grad)
    return records


@torch.no_grad()
def measure(model: nn.Module) -> list[dict]:
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding, SigmaLinear)):
            continue
        weight = module.weight.detach().float()
        singular = torch.linalg.svdvals(weight)
        nonzero = singular[singular > 0]
        probabilities = nonzero / nonzero.sum()
        erank = float(torch.exp(-(probabilities * probabilities.log()).sum())) if len(nonzero) else 0.0
        rows.append(dict(name=name, shape=list(weight.shape), effective_rank=erank,
                         full_rank=min(weight.shape), transformer=name.startswith("transformer.h.")))
    return rows
