from unittest.mock import patch

import torch

from optim.muon import Muon, zeropower_via_newtonschulz5
from optim.muon_spectral_L1_reg import MuonSpectralL1Reg


def test_pre_update_decoupled_direction_uses_pre_muon_weights() -> None:
    initial = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = torch.ones_like(parameter)
    lr = 0.02
    coefficient = 0.7
    optimizer = MuonSpectralL1Reg(
        [parameter],
        lr=lr,
        spectral_l1_reg_coef=coefficient,
        decoupled_pre_update=True,
    )
    muon_delta = torch.full_like(initial, -0.1)

    with patch.object(Muon, "step", side_effect=lambda: parameter.data.add_(muon_delta)):
        optimizer.step()

    expected = initial + muon_delta - lr * coefficient * (
        zeropower_via_newtonschulz5(initial, 5)
    )
    torch.testing.assert_close(parameter, expected)


def test_default_direction_uses_post_muon_weights() -> None:
    initial = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = torch.ones_like(parameter)
    lr = 0.02
    coefficient = 0.7
    optimizer = MuonSpectralL1Reg(
        [parameter], lr=lr, spectral_l1_reg_coef=coefficient
    )
    muon_delta = torch.full_like(initial, -0.1)

    with patch.object(Muon, "step", side_effect=lambda: parameter.data.add_(muon_delta)):
        optimizer.step()

    post_muon = initial + muon_delta
    expected = post_muon - lr * coefficient * (
        zeropower_via_newtonschulz5(post_muon, 5)
    )
    torch.testing.assert_close(parameter, expected)


def test_pre_update_decoupled_rejects_svt() -> None:
    parameter = torch.nn.Parameter(torch.eye(2))

    try:
        MuonSpectralL1Reg(
            [parameter], decoupled_pre_update=True, svt_interval=1
        )
    except ValueError as error:
        assert "does not support svt_interval" in str(error)
    else:
        raise AssertionError("pre-update decoupled regularization must reject SVT")
