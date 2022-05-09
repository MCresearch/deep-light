# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pl0 = np.loadtxt('p_0_pl1.dat',comments='#')
pl1 = np.loadtxt('p_1_pl1.dat',comments='#')
pl2 = np.loadtxt('p_2_pl1.dat',comments='#')
diff_pl = pl0   - pl1
#print(diff_zernike_coeff)
print("pl的差为：")
print(np.linalg.norm(diff_pl,ord=2))

plt.figure(1, dpi = 300)
plt.contourf(pl0)
plt.savefig("pl0.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(pl1)
plt.savefig("pl1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(pl2)
plt.savefig("pl2.png")
plt.close()
