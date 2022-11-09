# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
dl_inIntensity = np.loadtxt('dl_inIntensity.dat',comments='#')

plt.figure(1, dpi = 300)
plt.contourf(dl_inIntensity)
plt.savefig("./pictures/dl_inIntensity.png")
plt.close()


dl_inPhase_all = np.loadtxt('dl_inPhase_1.239100.dat',comments='#')
dl_inPhase = dl_inPhase_all[0:32,]


plt.figure(1, dpi = 300)
plt.contourf(dl_inPhase)
plt.savefig("./pictures/dl_inPhase.png")
plt.close()

dl_focusing_all = np.loadtxt('dl_focusing_1.239100.dat',comments='#')
dl_focusing = dl_focusing_all[0:32,]
dl_focusing_ur = dl_focusing_all[32:64,]
dl_focusing_ui = dl_focusing_all[64:96,]


plt.figure(1, dpi = 300)
plt.contourf(dl_focusing)
plt.savefig("./pictures/dl_focusing.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_focusing_ur)
plt.savefig("./pictures/dl_focusing_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_focusing_ui)
plt.savefig("./pictures/dl_focusing_ui.png")
plt.close()


dl_mdfph1_all = np.loadtxt('dl_mdfph1_1.239100.dat',comments='#')
dl_mdfph1 = dl_mdfph1_all[0:32,]
dl_mdfph1_ur = dl_mdfph1_all[32:64,]
dl_mdfph1_ui = dl_mdfph1_all[64:96,]


plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph1)
plt.savefig("./pictures/dl_mdfph1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph1_ur)
plt.savefig("./pictures/dl_mdfph1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph1_ui)
plt.savefig("./pictures/dl_mdfph1_ui.png")
plt.close()



dl_my_fft2d1_all = np.loadtxt('dl_my_fft2d1_1.239100.dat',comments='#')
dl_my_fft2d1 = dl_my_fft2d1_all[0:32,]
dl_my_fft2d1_ur = dl_my_fft2d1_all[32:64,]
dl_my_fft2d1_ui = dl_my_fft2d1_all[64:96,]



plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1)
plt.savefig("./pictures/dl_my_fft2d1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1_ur)
plt.savefig("./pictures/dl_my_fft2d1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1_ui)
plt.savefig("./pictures/dl_my_fft2d1_ui.png")
plt.close()


dl_evol1_all = np.loadtxt('dl_evol1_1.239100.dat',comments='#')
dl_evol1 = dl_evol1_all[0:32,]
dl_evol1_ur = dl_evol1_all[32:64,]
dl_evol1_ui = dl_evol1_all[64:96,]



plt.figure(1, dpi = 300)
plt.contourf(dl_evol1)
plt.savefig("./pictures/dl_evol1.png")
plt.close()


plt.figure(1, dpi = 300)
plt.contourf(dl_evol1_ur)
plt.savefig("./pictures/dl_evol1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_evol1_ui)
plt.savefig("./pictures/dl_evol1_ui.png")
plt.close()



dl_my_fft2d2_all = np.loadtxt('dl_my_fft2d2_1.239100.dat',comments='#')
dl_my_fft2d2 = dl_my_fft2d1_all[0:32,]
dl_my_fft2d2_ur = dl_my_fft2d2_all[32:64,]
dl_my_fft2d2_ui = dl_my_fft2d2_all[64:96,]


plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d2)
plt.savefig("./pictures/dl_my_fft2d2.png")
plt.close()


plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d2_ur)
plt.savefig("./pictures/dl_my_fft2d2_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d2_ui)
plt.savefig("./pictures/dl_my_fft2d2_ui.png")
plt.close()

dl_mdfph2_all = np.loadtxt('dl_mdfph2_1.239100.dat',comments='#')
dl_mdfph2 = dl_mdfph1_all[0:32,]
dl_mdfph2_ur = dl_mdfph1_all[32:64,]
dl_mdfph2_ui = dl_mdfph1_all[64:96,]


plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph2)
plt.savefig("./pictures/dl_mdfph2.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph2_ur)
plt.savefig("./pictures/dl_mdfph2_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph2_ui)
plt.savefig("./pictures/dl_mdfph2_ui.png")
plt.close()
'''
dl_outIntensity_all = np.loadtxt('dl_outIntensity.dat',comments='#')
dl_outIntensity = dl_outIntensity_all[320:352,]

plt.figure(1, dpi = 300)
plt.contourf(dl_outIntensity)
plt.savefig("./pictures/dl_outIntensity.png")
plt.close()





