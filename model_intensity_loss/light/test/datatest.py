# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")

outIntensity = np.loadtxt('./aznk_7_4dl_outIntensity.dat')
dl_outIntensity = np.load('../data/far_field_intens_orig.npy')
dl_outIntensity1 = dl_outIntensity[0,:,:]
# 相对误差
diff_outIntensity = np.average(np.abs(outIntensity-dl_outIntensity1)/outIntensity,axis=0)
np.savetxt("diff_outIntensity",diff_outIntensity)
print("outIntensity 的差为：")
print(np.linalg.norm(diff_outIntensity,ord=2) )

plt.figure(1, dpi = 300)
plt.contourf(outIntensity)
plt.savefig("outIntensity.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_outIntensity1)
plt.savefig("dl_outIntensity.png")
plt.close()






