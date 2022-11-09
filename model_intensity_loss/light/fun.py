
import numpy as np
def prop1(n_grid,n1,dz,kp,aa,h):
    tt = 0.0
    t0 = 0.0
    j1 = 0
    tt = dz / (2 * kp) * pow((2 * np.pi / aa), 2)
    for j in range(n_grid):
        j1 = j + 1 - n1
        t0 = tt * j1 * j1
        h[j] = -t0


def evol1(n_grid, h, img0_):
    cx = 0.0
    for i in range(n_grid):
        for j in range(n_grid):
            img0_[i,j] = img0_[i,j]*np.exp(1j*h[i])
            img0_[i,j] = img0_[i,j]*np.exp(1j*h[j])
    