"""Check the curve chain rule and the unchanged endpoint weights."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/cloud'))
from connectivity_experiment import add_path_regularizer, interpolate
from optim.adamw_spectral_L1_reg import zeropower_via_newtonschulz5

class SpectralPathTest(unittest.TestCase):
    def test_chain_rule(self):
        torch.manual_seed(3)
        left = {'embedding': torch.randn(8, 3), 'norm': torch.randn(3)}
        right = {k: torch.randn_like(v) for k,v in left.items()}
        before = {k:v.clone() for k,v in left.items()}
        for t in (0., .2, .5, .9, 1.):
            control = {k: torch.nn.Parameter(torch.randn_like(v)) for k,v in left.items()}
            for p in control.values():
                p.grad = torch.zeros_like(p)
            weights = interpolate(left, right, control, t)
            expected = 2*t*(1-t)*(t*.7)*zeropower_via_newtonschulz5(weights['embedding'], 5)
            stats = add_path_regularizer(left, right, control, t, 0., .7)
            torch.testing.assert_close(control['embedding'].grad, expected)
            torch.testing.assert_close(control['norm'].grad, torch.zeros(3))
            self.assertAlmostEqual(stats['spectral_lambda'], t*.7)
            for k in left:
                torch.testing.assert_close(left[k], before[k], rtol=0, atol=0)

if __name__ == '__main__':
    unittest.main()
