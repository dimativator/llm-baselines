"""Numerical and optimizer integration checks before allocating training GPUs."""
import importlib.util
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[2] / 'src/optim/adamw_schatten4.py'
spec = importlib.util.spec_from_file_location('schatten4', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(0)
torch.set_num_threads(2)
for shape in ((7, 4), (4, 7), (4, 4)):
    weight = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    singular = torch.linalg.svdvals(weight)
    penalty = min(shape) ** 0.5 / 2 * singular.pow(4).sum().sqrt()
    expected, = torch.autograd.grad(penalty, weight)
    torch.testing.assert_close(module.schatten4_gradient(weight), expected, rtol=1e-10, atol=1e-10)
    flat = torch.zeros(shape, dtype=torch.float64)
    for index in range(min(shape)):
        flat[index, index] = 2.0
    torch.testing.assert_close(module.schatten4_gradient(flat), flat)
    assert torch.equal(module.schatten4_gradient(torch.zeros_like(flat)), torch.zeros_like(flat))

weight = torch.nn.Parameter(torch.randn(7, 4, dtype=torch.float64))
exempt = torch.nn.Parameter(torch.randn(7, 4, dtype=torch.float64))
reference = [torch.nn.Parameter(p.detach().clone()) for p in (weight, exempt)]
optimizer = module.AdamWSchatten4([{'params': [weight], 'weight_decay': .3},
                                  {'params': [exempt], 'weight_decay': 0.0}], lr=.002)
adam = torch.optim.AdamW(reference, lr=.002, weight_decay=0.0)
for _ in range(3):
    for p, q in zip((weight, exempt), reference):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    adam.step()
    expected = reference[0].detach() - .002 * .3 * module.schatten4_gradient(reference[0].detach())
    optimizer.step()
    torch.testing.assert_close(weight, expected)
    torch.testing.assert_close(exempt, reference[1])
    for key in ('exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(optimizer.state[weight][key], adam.state[reference[0]][key])
    reference[0].data.copy_(weight)
assert optimizer.param_groups[0]['weight_decay'] == .3
for shape in ((32, 64), (768, 768)):
    zero = torch.zeros(shape)
    assert torch.equal(module.schatten4_gradient(zero), zero)
print('SCHATTEN4_NUMERICAL_TESTS_PASSED', flush=True)
