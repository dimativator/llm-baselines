import torch

from optim.adamw_spectral_L1_reg import (
    AdamWSpectralL1Reg,
    zeropower_via_newtonschulz5,
)


def test_coupled_spectral_regularizer_enters_adam_moments() -> None:
    initial = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
    task_gradient = torch.full_like(initial, -0.01)
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = task_gradient.clone()

    lr = 0.01
    coefficient = 0.7
    optimizer = AdamWSpectralL1Reg(
        [parameter],
        lr=lr,
        betas=(0.0, 0.0),
        eps=1e-8,
        spectral_l1_reg_coef=coefficient,
        coupled=True,
    )

    expected_gradient = task_gradient + coefficient * zeropower_via_newtonschulz5(
        initial, 5
    )
    expected_parameter = initial - lr * expected_gradient / (
        expected_gradient.abs() + 1e-8
    )

    optimizer.step()

    torch.testing.assert_close(parameter.grad, task_gradient)
    torch.testing.assert_close(optimizer.state[parameter]["first_momentum"], expected_gradient)
    torch.testing.assert_close(parameter, expected_parameter)


def test_coupled_spectral_regularizer_rejects_svt() -> None:
    parameter = torch.nn.Parameter(torch.eye(2))

    try:
        AdamWSpectralL1Reg([parameter], coupled=True, svt_interval=1)
    except ValueError as error:
        assert "does not support svt_interval" in str(error)
    else:
        raise AssertionError("coupled spectral regularization must reject SVT")


def test_pre_update_decoupled_spectral_regularizer_matches_slorr_order() -> None:
    initial = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
    task_gradient = torch.tensor([[0.2, -0.4], [0.1, -0.3]])
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = task_gradient.clone()

    lr = 0.01
    coefficient = 0.7
    optimizer = AdamWSpectralL1Reg(
        [parameter],
        lr=lr,
        betas=(0.0, 0.0),
        eps=1e-8,
        spectral_l1_reg_coef=coefficient,
        decoupled_pre_update=True,
    )

    pre_update_direction = zeropower_via_newtonschulz5(initial, 5)
    adam_update = task_gradient / (task_gradient.abs() + 1e-8)
    expected_parameter = (
        initial - lr * adam_update - lr * coefficient * pre_update_direction
    )

    optimizer.step()

    torch.testing.assert_close(
        optimizer.state[parameter]["first_momentum"], task_gradient
    )
    torch.testing.assert_close(parameter, expected_parameter)


def test_default_spectral_regularizer_remains_post_update() -> None:
    initial = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
    task_gradient = torch.tensor([[0.2, -0.4], [0.1, -0.3]])
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = task_gradient.clone()

    lr = 0.01
    coefficient = 0.7
    optimizer = AdamWSpectralL1Reg(
        [parameter],
        lr=lr,
        betas=(0.0, 0.0),
        eps=1e-8,
        spectral_l1_reg_coef=coefficient,
    )

    adam_update = task_gradient / (task_gradient.abs() + 1e-8)
    post_task_weights = initial - lr * adam_update
    expected_parameter = post_task_weights - lr * coefficient * (
        zeropower_via_newtonschulz5(post_task_weights, 5)
    )

    optimizer.step()

    torch.testing.assert_close(parameter, expected_parameter)


def test_pre_update_decoupled_spectral_regularizer_rejects_svt() -> None:
    parameter = torch.nn.Parameter(torch.eye(2))

    try:
        AdamWSpectralL1Reg(
            [parameter], decoupled_pre_update=True, svt_interval=1
        )
    except ValueError as error:
        assert "does not support svt_interval" in str(error)
    else:
        raise AssertionError("pre-update decoupled regularization must reject SVT")


def test_pre_update_decoupled_and_coupled_are_exclusive() -> None:
    parameter = torch.nn.Parameter(torch.eye(2))

    try:
        AdamWSpectralL1Reg(
            [parameter], coupled=True, decoupled_pre_update=True
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("pre-update decoupled and coupled modes must be exclusive")
