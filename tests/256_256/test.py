# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


inIntensity = np.loadtxt('inIntensity.dat')
dl_inIntensity = np.loadtxt('dl_inIntensity.dat')
diff_inIntensity = inIntensity-dl_inIntensity
print("inIntensity 的差为：")
print(np.linalg.norm(diff_inIntensity,ord=2) )

plt.figure(1, dpi = 300)
plt.contourf(inIntensity)
plt.savefig("inIntensity.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_inIntensity)
plt.savefig("dl_inIntensity.png")
plt.close()


'''
zernike_coeff = np.loadtxt('zernike_coeff.dat')
dl_zernike_coeff = np.loadtxt('dl_zernike_coeff.dat')
diff_zernike_coeff = zernike_coeff  - dl_zernike_coeff
#print(diff_zernike_coeff)
print("zernike_coeff 的差为：")
print(np.linalg.norm(diff_zernike_coeff,ord=2))
'''
'''
zernike_cg = np.loadtxt('zernike_cg.dat')
dl_zernike_cg = np.loadtxt('dl_zernike_cg.dat')
diff_zernike_cg = zernike_cg - dl_zernike_cg
#nonzero = np.nonzero(diff_zernike_cg)
#print(diff_zernike_cg[nonzero])
f = open("1.txt","w")
yueyue = [0 for x in range(0, 90000)]
k = 0
for i in range(1136712):
  for j in range(2):
    if abs(diff_zernike_cg[i][j]) > 1e-5:
        yueyue[k] = zernike_cg[i][0]
        k = k+1
print(yueyue,file = f)

#print(diff_zernike_cg)
print("diff_zernike_cg的差为：")
print(np.linalg.norm(diff_zernike_cg,ord=2) )
'''


inPhase = np.loadtxt('inPhase.dat')
dl_inPhase = np.loadtxt('dl_inPhase.dat')
diff_inPhase = inPhase-dl_inPhase
print("inPhase 的差为：")
print(np.linalg.norm(diff_inPhase,ord=2) )


inPhase_intensity = np.loadtxt('inPhase_inIntensity.dat')
dl_inPhase_intensity = np.loadtxt('dl_inPhase_intensity.dat')
diff_inPhase_intensity = inPhase_intensity-dl_inPhase_intensity
print("inPhase_intensity 的差为：")
print(np.linalg.norm(diff_inPhase_intensity,ord=2) )

'''
fft_initialize = np.loadtxt('fft_initialize.dat')
dl_fft_initialize = np.loadtxt('dl_fft_initialize.dat')
diff_fft_initialize = fft_initialize-dl_fft_initialize
print("fft_initialize 的差为：")
print(np.linalg.norm(diff_fft_initialize,ord=2) )
'''

focusing = np.loadtxt('focusing_inIntensity.dat')
dl_focusing = np.loadtxt('dl_focusing.dat')
diff_focusing = focusing-dl_focusing
print("focusing 的差为：")
print(np.linalg.norm(diff_focusing,ord=2) )

plt.figure(1, dpi = 300)
plt.contourf(focusing)
plt.savefig("focusing.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_focusing)
plt.savefig("dl_focusing.png")
plt.close()


mdfph1 = np.loadtxt('mdfph1_inIntensity.dat')
dl_mdfph1 = np.loadtxt('dl_mdfph1.dat')
diff_mdfph1 = mdfph1-dl_mdfph1
print("mdfph1 的差为：")
print(np.linalg.norm(diff_mdfph1,ord=2) )

plt.figure(1, dpi = 300)
plt.contourf(mdfph1)
plt.savefig("mdfph1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph1)
plt.savefig("dl_mdfph1.png")
plt.close()


my_fft2d1 = np.loadtxt('my_fft2d1_inIntensity.dat')
dl_my_fft2d1 = np.loadtxt('dl_my_fft2d1.dat')
diff_my_fft2d1 = my_fft2d1 - dl_my_fft2d1
print("my_fft2d1 的差为：")
print(np.linalg.norm(diff_my_fft2d1,ord=2) )
np.max(my_fft2d1)

plt.figure(1, dpi = 300)
plt.contourf(my_fft2d1)
plt.savefig("my_fft2d1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1)
plt.savefig("dl_my_fft2d1.png")
plt.close()

dl_mdfph1_ur = np.loadtxt('dl_mdfph1_ur.dat')
dl_mdfph1_ui = np.loadtxt('dl_mdfph1_ui.dat')
dl_focusing_ur = np.loadtxt('dl_focusing_ur.dat')
dl_focusing_ui = np.loadtxt('dl_focusing_ui.dat')

plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph1_ur)
plt.savefig("dl_mdfph1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_mdfph1_ui)
plt.savefig("dl_mdfph1_ui.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_focusing_ur)
plt.savefig("dl_focusing_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_focusing_ui)
plt.savefig("dl_focusing_ui.png")
plt.close()





prop1 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/prop1_inIntensity.dat')
dl_prop1 = np.loadtxt('dl_prop1.dat')
diff_prop1 = prop1-dl_prop1
print("prop1 的差为：")
print(np.linalg.norm(diff_prop1,ord=2) )

evol1 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/evol1_inIntensity.dat')
dl_evol1 = np.loadtxt('dl_evol1.dat')
diff_evol1 = evol1-dl_evol1
print("evol1 的差为：")
print(np.linalg.norm(diff_evol1,ord=2) )

my_fft2d2 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d2_inIntensity.dat')
dl_my_fft2d2= np.loadtxt('dl_my_fft2d2.dat')
diff_my_fft2d2 = my_fft2d2-dl_my_fft2d2
print("my_fft2d2 的差为：")
print(np.linalg.norm(diff_my_fft2d2,ord=2) )

plt.figure(1, dpi = 300)
plt.contourf(my_fft2d2)
plt.savefig("my_fft2d2.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d2)
plt.savefig("dl_my_fft2d2.png")
plt.close()


mdfph2 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/mdfph2_inIntensity.dat')
dl_mdfph2 = np.loadtxt('dl_mdfph2.dat')
diff_mdfph2 = mdfph2-dl_mdfph2
print("mdfph2 的差为：")
print(np.linalg.norm(diff_mdfph2,ord=2) )


outIntensity = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/outIntensity.dat')
dl_outIntensity = np.loadtxt('dl_outIntensity.dat')
diff_outIntensity = outIntensity-dl_outIntensity
print("outIntensity 的差为：")
print(np.linalg.norm(diff_outIntensity,ord=2) )



