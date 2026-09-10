import math
import torch


def zeropower_via_newtonschulz5(W, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of W. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = W is the SVD.
    """
    # assert len(W.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = W
    if W.size(0) > W.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )
        X = a * X + B @ X

    if W.size(0) > W.size(1):
        X = X.T
    return X


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


class AdamWSpectralL1Reg(torch.optim.Optimizer):
    """AdamW with post-step, pre-update decoupled, or coupled spectral L1 regularization.

    By default the nuclear-norm direction is computed from the *gradient-stepped*
    weights Z = W - lr*adam_update. Set ``decoupled_pre_update=True`` for the
    AdamW/SLORR-style decoupled update: compute the nuclear-norm direction from
    the pre-update weights W, apply the Adam task update, then subtract the saved
    direction. Set ``coupled=True`` to instead add the nuclear-norm subgradient
    to the task gradient before Adam updates its moments.

    On most steps the prox is approximated by the cheap Newton-Schulz subgradient
    step (subtract tau*U V^T, tau = lr*spectral_l1_reg_coef). Every ``svt_interval``
    steps it is instead computed *exactly* via singular-value soft-thresholding
    (SVT), which zeros sub-threshold singular values and gives real rank reduction.
    Set ``svt_interval=0`` to disable SVT and use Newton-Schulz on every step.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        spectral_l1_reg_coef=0.1,
        svt_interval=0,
        svt_thresh=None,
        coupled=False,
        decoupled_pre_update=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if svt_interval < 0:
            raise ValueError("svt_interval must be >= 0")
        if coupled and svt_interval != 0:
            raise ValueError(
                "coupled spectral L1 regularization does not support svt_interval; "
                "SVT is a decoupled proximal step"
            )
        if coupled and decoupled_pre_update:
            raise ValueError(
                "coupled and decoupled_pre_update spectral regularization are "
                "mutually exclusive"
            )
        if decoupled_pre_update and svt_interval != 0:
            raise ValueError(
                "decoupled_pre_update spectral regularization does not support "
                "svt_interval; SVT acts on the post-update weights"
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            spectral_l1_reg_coef=spectral_l1_reg_coef,
            svt_interval=svt_interval,
            svt_thresh=svt_thresh,
            coupled=coupled,
            decoupled_pre_update=decoupled_pre_update,
        )

        super(AdamWSpectralL1Reg, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWSpectralL1Reg, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            spectral_l1_reg_coef = group["spectral_l1_reg_coef"]
            svt_interval = group["svt_interval"]
            svt_thresh = group["svt_thresh"]
            coupled = group["coupled"]
            decoupled_pre_update = group["decoupled_pre_update"]

            for p in group["params"]:
                grad = p.grad

                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported!")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["first_momentum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["second_momentum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m = state["first_momentum"]
                v = state["second_momentum"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Coupled spectral WD adds the nuclear-norm subgradient to the
                # task gradient before Adam updates either moment. ``grad`` is
                # deliberately not modified so logged task loss and accumulated
                # task gradients remain separate from the regularizer.
                orig_shape = p.data.shape
                is_conv = len(orig_shape) == 4
                adam_grad = grad
                pre_update_spectral_direction = None
                if (
                    coupled
                    and (len(orig_shape) == 2 or is_conv)
                    and spectral_l1_reg_coef > 0
                ):
                    weight = p.data.view(orig_shape[0], -1) if is_conv else p.data
                    nuclear_subgradient = zeropower_via_newtonschulz5(weight, 5)
                    adam_grad = grad.add(
                        nuclear_subgradient.view(orig_shape) if is_conv else nuclear_subgradient,
                        alpha=spectral_l1_reg_coef,
                    )
                elif (
                    decoupled_pre_update
                    and (len(orig_shape) == 2 or is_conv)
                    and spectral_l1_reg_coef > 0
                ):
                    weight = p.data.view(orig_shape[0], -1) if is_conv else p.data
                    pre_update_spectral_direction = zeropower_via_newtonschulz5(
                        weight, 5
                    )

                # --- AdamW gradient step: p.data -> Z = W - lr * adam_update ---
                # Decay the first and second moment running average coefficient
                m.lerp_(adam_grad, 1 - beta1)
                v.mul_(beta2).addcmul_(adam_grad, adam_grad, value=(1 - beta2))

                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.data.addcdiv_(m, denom, value=-(lr / bias_correction1))

                # AdamW/SLORR-style decoupling: the regularizer direction was
                # computed from W before the task update, but is applied after it.
                if pre_update_spectral_direction is not None:
                    p.data.add_(
                        pre_update_spectral_direction.view(orig_shape)
                        if is_conv
                        else pre_update_spectral_direction,
                        alpha=-(lr * spectral_l1_reg_coef),
                    )

                # --- Nuclear-norm prox applied to the stepped weights Z ---
                # (faithful to W_{k+1} = prox(W_k - lr*grad)). Conv2d weights
                # [out_ch, in_ch, kh, kw] are viewed as the standard "filter
                # matrix" [out_ch, in_ch*kh*kw] so the same 2D nuclear-norm
                # prox applies to them too, unmodified otherwise.
                if (
                    not coupled
                    and not decoupled_pre_update
                    and (len(orig_shape) == 2 or is_conv)
                    and spectral_l1_reg_coef > 0
                ):
                    W = p.data.view(orig_shape[0], -1) if is_conv else p.data
                    do_svt = (
                        svt_interval > 0 and state["step"] % svt_interval == 0
                    )
                    if do_svt:
                        # Exact prox: singular-value soft-thresholding (zeros the tail).
                        tau = (
                            svt_thresh
                            if svt_thresh is not None
                            else lr * spectral_l1_reg_coef
                        )
                        new_W = _singular_value_threshold(W, tau)
                        p.data.copy_(new_W.view(orig_shape) if is_conv else new_W)
                    else:
                        l1_weight_reg = zeropower_via_newtonschulz5(W, 5)
                        p.data.add_(
                            l1_weight_reg.view(orig_shape) if is_conv else l1_weight_reg,
                            alpha=-(lr * spectral_l1_reg_coef),
                        )
                elif len(orig_shape) not in (2, 4):
                    p.data.mul_(1 - lr * wd)

        return loss
