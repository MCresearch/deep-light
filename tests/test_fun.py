import unittest
from context import prop1, evol1, fftt, fftt2
import numpy as np
import torch

class FunTest(unittest.TestCase):
    def test_prop1(self):
        n_grid = 8
        n1 = 5
        dz = 0.2
        kp = 4
        aa = 2
        h = np.zeros(n_grid)
        ans_ref = np.array([-3.94784176, -2.22066099, -0.98696044, -0.24674011, -0., -0.24674011, -0.98696044, -2.22066099])
        prop1(n_grid, n1, dz, kp, aa, h)
        np.testing.assert_almost_equal(ans_ref, h)

    def test_evol1(self):
        n_grid = 3
        h = np.arange(0, 3, 1)*np.pi/2
        h = torch.tensor(h)
        img0_ = torch.complex(torch.ones((n_grid, n_grid)), imag=torch.zeros((n_grid, n_grid)))
        evol1(n_grid, h, img0_)
        img0_ = img0_.numpy()
        ans_ref = np.array([[1+0j, 1j, -1+0j], [1j, -1+0j, -1j], [-1+0j, -1j, 1+0j]])
        np.testing.assert_almost_equal(ans_ref, img0_)
    


if __name__ == "__main__":
    unittest.main()
