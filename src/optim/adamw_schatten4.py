"""Adam followed by decay for sqrt(r)/2 * ||W||_S4^2."""
import math

import torch


def schatten4_gradient(weight: torch.Tensor) -> torch.Tensor:
    """Exact gradient using the smaller Gram matrix; FP32 or FP64 arithmetic."""
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=weight.device.type, enabled=False):
        matrix = weight.to(dtype)
        if matrix.shape[0] >= matrix.shape[1]:
            gram = matrix.T @ matrix
            direction = matrix @ gram
        else:
            gram = matrix @ matrix.T
            direction = gram @ matrix
        scale = gram.square().sum().sqrt().clamp_min(torch.finfo(dtype).tiny)
        return direction.div_(scale).mul_(math.sqrt(min(matrix.shape)))


class AdamWSchatten4(torch.optim.AdamW):
    """Keep Adam moments free of regularization; apply Schatten-4 decay to Z."""

    @torch.no_grad()
    def step(self, closure=None):
        decay = [group['weight_decay'] for group in self.param_groups]
        try:
            for group in self.param_groups:
                group['weight_decay'] = 0.0
            loss = super().step(closure)
        finally:
            for group, coefficient in zip(self.param_groups, decay):
                group['weight_decay'] = coefficient
        for group in self.param_groups:
            coefficient = group['weight_decay']
            if not coefficient:
                continue
            for weight in group['params']:
                if weight.grad is None:
                    continue
                if weight.ndim != 2:
                    raise ValueError('Schatten-4 decay requires matrix decay groups')
                direction = schatten4_gradient(weight)
                weight.add_(direction, alpha=-group['lr'] * coefficient)
        return loss
