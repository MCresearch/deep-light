# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dl_inPhase = np.loadtxt('dl_inPhase.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_inPhase)
plt.savefig("dl_inPhase.png")
plt.close()

dl_inPhase_intensity = np.loadtxt('dl_inPhase_intensity.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_inPhase_intensity)
plt.savefig("dl_inPhase_intensity.png")
plt.close()


dl_focusing = np.loadtxt('dl_focusing.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_focusing)
plt.savefig("dl_focusing.png")
plt.close()

dl_mdfph1 = np.loadtxt('dl_mdfph1.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph1)
plt.savefig("dl_mdfph1.png")
plt.close()

dl_my_fft2d1 = np.loadtxt('dl_my_fft2d1.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1)
plt.savefig("dl_my_fft2d1.png")
plt.close()



dl_evol1 = np.loadtxt('dl_evol1.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_evol1)
plt.savefig("dl_evol1.png")
plt.close()

dl_my_fft2d2= np.loadtxt('dl_my_fft2d2.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d2)
plt.savefig("dl_my_fft2d2.png")
plt.close()

dl_mdfph2 = np.loadtxt('dl_mdfph2.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph2)
plt.savefig("dl_mdfph2.png")
plt.close()

dl_outIntensity = np.loadtxt('dl_outIntensity.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_outIntensity)
plt.savefig("dl_outIntensity.png")
plt.close()


