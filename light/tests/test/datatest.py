# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


outIntensity = np.loadtxt('./fortran/outIntensity.dat',comments='#')
dl_outIntensity = np.loadtxt('dl_outIntensity.dat',comments='#')
# 相对误差
diff_outIntensity = np.average(np.abs(outIntensity-dl_outIntensity)/outIntensity,axis=0)
np.savetxt("diff_outIntensity",diff_outIntensity)
print("outIntensity 的差为：")
print(np.linalg.norm(diff_outIntensity,ord=2) )

plt.figure(1, dpi = 300)
plt.contourf(outIntensity)
plt.savefig("outIntensity.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_outIntensity)
plt.savefig("dl_outIntensity.png")
plt.close()






