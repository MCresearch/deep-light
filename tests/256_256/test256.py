# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


inIntensity = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/inIntensity.dat',comments='#')
dl_inIntensity = np.loadtxt('dl_inIntensity.dat',comments='#')

plt.figure(1, dpi = 300)
plt.contourf(inIntensity)
plt.savefig("./pictures/inIntensity.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_inIntensity)
plt.savefig("./pictures/dl_inIntensity.png")
plt.close()


inPhase = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/inPhase.dat')
dl_inPhase_all = np.loadtxt('dl_inPhase_1.239100.dat',comments='#')
dl_inPhase = dl_inPhase_all[0:256,]

plt.figure(1, dpi = 300)
plt.contourf(inPhase)
plt.savefig("./pictures/inPhase.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_inPhase)
plt.savefig("./pictures/dl_inPhase.png")
plt.close()


focusing = np.loadtxt('focusing_inIntensity.dat')
dl_focusing_all = np.loadtxt('dl_focusing_1.239100.dat',comments='#')
dl_focusing = dl_focusing_all[0:256,]
dl_focusing_ur = dl_focusing_all[256:512,]
dl_focusing_ui = dl_focusing_all[512:768,]

plt.figure(1, dpi = 300)
plt.contourf(focusing)
plt.savefig("./pictures/focusing.png")
plt.close()

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


mdfph1 = np.loadtxt('mdfph1_inIntensity.dat')
dl_mdfph1_all = np.loadtxt('dl_mdfph1_1.239100.dat',comments='#')
dl_mdfph1 = dl_mdfph1_all[0:256,]
dl_mdfph1_ur = dl_mdfph1_all[256:512,]
dl_mdfph1_ui = dl_mdfph1_all[512:768,]


plt.figure(1, dpi = 300)
plt.contourf(mdfph1)
plt.savefig("./pictures/mdfph1.png")
plt.close()

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




my_fft2d1 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d1_inIntensity.dat')
my_fft2d1_ur = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d1_ur.dat')
my_fft2d1_ui = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d1_ui.dat')
dl_my_fft2d1_all = np.loadtxt('dl_my_fft2d1_1.239100.dat',comments='#')
dl_my_fft2d1 = dl_my_fft2d1_all[0:256,]
dl_my_fft2d1_ur = dl_my_fft2d1_all[256:512,]
dl_my_fft2d1_ui = dl_my_fft2d1_all[512:768,]


plt.figure(1, dpi = 300)
plt.contourf(my_fft2d1)
plt.savefig("./pictures/my_fft2d1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1)
plt.savefig("./pictures/dl_my_fft2d1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(my_fft2d1_ur)
plt.savefig("./pictures/my_fft2d1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1_ur)
plt.savefig("./pictures/dl_my_fft2d1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(my_fft2d1_ui)
plt.savefig("./pictures/my_fft2d1_ui.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1_ui)
plt.savefig("./pictures/dl_my_fft2d1_ui.png")
plt.close()


evol1 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/evol1_inIntensity.dat')
evol1_ur = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/evol1_ur.dat')
evol1_ui = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/evol1_ui.dat')
dl_evol1_all = np.loadtxt('dl_evol1_1.239100.dat',comments='#')
dl_evol1 = dl_evol1_all[0:256,]
dl_evol1_ur = dl_evol1_all[256:512,]
dl_evol1_ui = dl_evol1_all[512:768,]

plt.figure(1, dpi = 300)
plt.contourf(evol1)
plt.savefig("./pictures/evol1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_evol1)
plt.savefig("./pictures/dl_evol1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(evol1_ur)
plt.savefig("./pictures/evol1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_evol1_ur)
plt.savefig("./pictures/dl_evol1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(evol1_ui)
plt.savefig("./pictures/evol1_ui.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_evol1_ui)
plt.savefig("./pictures/dl_evol1_ui.png")
plt.close()



my_fft2d2 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d2_inIntensity.dat')
my_fft2d2_ur = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d2_ur.dat')
my_fft2d2_ui = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d2_ui.dat')
dl_my_fft2d2_all = np.loadtxt('dl_my_fft2d2_1.239100.dat',comments='#')
dl_my_fft2d2 = dl_my_fft2d1_all[0:256,]
dl_my_fft2d2_ur = dl_my_fft2d2_all[256:512,]
dl_my_fft2d2_ui = dl_my_fft2d2_all[512:768,]


plt.figure(1, dpi = 300)
plt.contourf(my_fft2d2)
plt.savefig("./pictures/my_fft2d2.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d2)
plt.savefig("./pictures/dl_my_fft2d2.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(my_fft2d2_ur)
plt.savefig("./pictures/my_fft2d2_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d2_ur)
plt.savefig("./pictures/dl_my_fft2d2_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(my_fft2d2_ui)
plt.savefig("./pictures/my_fft2d2_ui.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d2_ui)
plt.savefig("./pictures/dl_my_fft2d2_ui.png")
plt.close()


mdfph2 = np.loadtxt('mdfph2_inIntensity.dat')
dl_mdfph2_all = np.loadtxt('dl_mdfph2_1.239100.dat',comments='#')
dl_mdfph2 = dl_mdfph1_all[0:256,]
dl_mdfph2_ur = dl_mdfph1_all[256:512,]
dl_mdfph2_ui = dl_mdfph1_all[512:768,]


plt.figure(1, dpi = 300)
plt.contourf(mdfph2)
plt.savefig("./pictures/mdfph2.png")
plt.close()

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


outIntensity = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/outIntensity.dat')
dl_outIntensity_all = np.loadtxt('dl_outIntensity_1.239100.dat',comments='#')
dl_outIntensity = dl_outIntensity_all[0:256,]


plt.figure(1, dpi = 300)
plt.contourf(outIntensity)
plt.savefig("./pictures/outIntensity.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_outIntensity)
plt.savefig("./pictures/dl_outIntensity.png")
plt.close()





