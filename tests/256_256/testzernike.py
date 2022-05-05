# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

zernike_coeff = np.loadtxt('zernike_coeff.dat')
dl_zernike_coeff = np.loadtxt('dl_zernike_coeff_0.239100.dat',comments='#')
diff_zernike_coeff = zernike_coeff  - dl_zernike_coeff
#print(diff_zernike_coeff)
print("zernike_coeff 的差为：")
print(np.linalg.norm(diff_zernike_coeff,ord=2))