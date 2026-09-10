"""Muon with post-step or pre-update decoupled nuclear-norm regularization."""

import math
import os

import torch
import torch.distributed as dist

from .muon import Muon, zeropower_via_newtonschulz5


def _singular_value_threshold(W, tau):
    """Exact proximal operator of the nuclear norm: singular-value soft-thresholding.

        prox_{tau ||.||_*}(W) = U * diag(max(sigma_i - tau, 0)) * V^T

    Unlike the Newton-Schulz subgradient step (which subtracts tau*U V^T and only
    shifts the whole spectrum), this sets singular values below ``tau`` to exactly
    zero, producing genuine rank reduction. Only the surviving components are kept.
    """
    orig_dtype = W.dtype
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    S = torch.clamp(S - tau, min=0.0)
    keep = S > 0
    if not torch.any(keep):
        return torch.zeros_like(W)
    U = U[:, keep]
    S = S[keep]
    Vh = Vh[keep, :]
    return ((U * S.unsqueeze(0)) @ Vh).to(orig_dtype)


class MuonSpectralL1Reg(Muon):
    """Muon with spectral L1 (nuclear-norm) regularization.

    By default, the nuclear-norm direction is computed from the post-Muon weights.
    With ``decoupled_pre_update=True``, it is computed from the pre-update weights
    and saved before applying the Muon task update, giving
    ``W <- W - lr * muon_update - lr * coef * NS(W_pre)``.

    On most steps, uses the cheap Newton-Schulz subgradient approximation:
        W <- W - tau * zeropower(W)
    Every ``svt_interval`` steps (if > 0), uses exact singular-value soft-thresholding (SVT):
        W <- prox_{tau||.||_*}(W)
    which zeros sub-threshold singular values and produces real rank reduction.

    Arguments:
        muon_params: Parameters to be optimized by Muon.
        lr: Muon learning rate (spectral norm of updates). (0.02 is a good default)
        momentum: SGD momentum used internally by Muon. (0.95 is a good default)
        nesterov: Whether to use Nesterov momentum. (recommended)
        ns_steps: Number of Newton-Schulz iterations. (6 is probably always enough)
        adamw_params: Parameters for AdamW fallback (1D, embeddings, head).
        adamw_lr: Learning rate for the internal AdamW.
        adamw_betas: Betas for the internal AdamW.
        adamw_eps: Epsilon for the internal AdamW.
        adamw_wd: Weight decay for the internal AdamW.
        spectral_l1_reg_coef: Nuclear-norm penalty strength.
        svt_interval: How often to do exact SVT. 0 = always use NS subgradient.
        svt_thresh: SVT threshold. If None, uses lr * spectral_l1_reg_coef.
        decoupled_pre_update: Compute the NS direction from pre-update weights.
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
        spectral_l1_reg_coef=0.1,
        svt_interval=0,
        svt_thresh=None,
        decoupled_pre_update=False,
    ):
        if decoupled_pre_update and svt_interval != 0:
            raise ValueError(
                "decoupled_pre_update spectral regularization does not support "
                "svt_interval; SVT acts on the post-update weights"
            )
        super().__init__(
            muon_params=muon_params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_params=adamw_params,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        self.spectral_l1_reg_coef = spectral_l1_reg_coef
        self.svt_interval = svt_interval
        self.svt_thresh = svt_thresh
        self.decoupled_pre_update = decoupled_pre_update
        self._global_step = 0

    @torch.no_grad()
    def step(self):
        coef = self.spectral_l1_reg_coef
        pre_update_directions = {}
        if self.decoupled_pre_update and coef > 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if self.state[p].get("use_muon", False) and p.ndim == 2:
                        pre_update_directions[p] = zeropower_via_newtonschulz5(
                            p.data, steps=5
                        )

        super().step()

        self._global_step += 1
        step = self._global_step

        if coef <= 0:
            return

        for group in self.param_groups:
            lr = group["lr"]
            tau = self.svt_thresh if self.svt_thresh is not None else lr * coef

            for p in group["params"]:
                # Only apply spectral L1 to 2D params that receive the Muon update
                if not self.state[p].get("use_muon", False):
                    continue
                if p.ndim != 2:
                    continue

                if self.decoupled_pre_update:
                    p.data.add_(pre_update_directions[p], alpha=-tau)
                elif self.svt_interval > 0 and step % self.svt_interval == 0:
                    p.data.copy_(_singular_value_threshold(p.data, tau))
                else:
                    ns_approx = zeropower_via_newtonschulz5(p.data, steps=5)
                    p.data.add_(ns_approx, alpha=-tau)
