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
plt.contourf(dl_idiff_inPhase = inPhase-dl_inPhase
print("inPhase 的差为：")
print(np.linalg.norm(diff_inPhase,ord=2) )
nIntensity)
plt.savefig("dl_inIntensity.png")
plt.close()



zernike_coeff = np.loadtxt('zernike_coeff.dat')
dl_zernike_coeff = np.loadtxt('dl_zernike_coeff.dat')
diff_zernike_coeff = zernike_coeff  - dl_zernike_coeff
#print(diff_zernike_coeff)
print("zernike_coeff 的差为：")
print(np.linalg.norm(diff_zernike_coeff,ord=2))

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


inPhase = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/inPhase.dat')
dl_inPhase = np.loadtxt('dl_inPhase.dat')
diff_inPhase = inPhase-dl_inPhase
print("inPhase 的差为：")
print(np.linalg.norm(diff_inPhase,ord=2) )

plt.figure(1, dpi = 300)
plt.contourf(inPhase)
plt.savefig("inPhase.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_inPhase)
plt.savefig("dl_inPhase.png")
plt.close()

'''
inPhase_intensity = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/inPhase_inIntensity.dat')
dl_inPhase_intensity = np.loadtxt('dl_inPhase_intensity.dat')
diff_inPhase_intensity = inPhase_intensity-dl_inPhase_intensity
print("inPhase_intensity 的差为：")
print(np.linalg.norm(diff_inPhase_intensity,ord=2) )
'''

'''
fft_initialize = np.loadtxt('fft_initialize.dat')
dl_fft_initialize = np.loadtxt('dl_fft_initialize.dat')
diff_fft_initialize = fft_initialize-dl_fft_initialize
print("fft_initialize 的差为：")
print(np.linalg.norm(diff_fft_initialize,ord=2) )
'''
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


my_fft2d1 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d1_inIntensity.dat')
my_fft2d1_ur = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d1_ur.dat')
my_fft2d1_ui = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/my_fft2d1_ui.dat')
dl_my_fft2d1 = np.loadtxt('dl_my_fft2d1.dat')
dl_my_fft2d1_ur = np.loadtxt('dl_my_fft2d1_ur.dat')
dl_my_fft2d1_ui = np.loadtxt('dl_my_fft2d1_ui.dat')
diff_my_fft2d1 = my_fft2d1 - dl_my_fft2d1
diff_my_fft2d1_ur = my_fft2d1_ur - dl_my_fft2d1_ur
diff_my_fft2d1_ui = my_fft2d1_ui - dl_my_fft2d1_ui
print("my_fft2d1 的差为：")
print(np.linalg.norm(diff_my_fft2d1,ord=2) )
print("my_fft2d1_ur 的差为：")
print(np.linalg.norm(diff_my_fft2d1_ur,ord=2) )
print("my_fft2d1_ui 的差为：")
print(np.linalg.norm(diff_my_fft2d1_ui,ord=2) )

plt.figure(1, dpi = 300)
plt.contourf(my_fft2d1)
plt.savefig("my_fft2d1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1)
plt.savefig("dl_my_fft2d1.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(my_fft2d1_ur)
plt.savefig("my_fft2d1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1_ur)
plt.savefig("dl_my_fft2d1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(my_fft2d1_ui)
plt.savefig("my_fft2d1_ui.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_my_fft2d1_ui)
plt.savefig("dl_my_fft2d1_ui.png")
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




prop1_hr = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/prop1_hr.dat')
prop1_hi = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/prop1_hi.dat')
dl_prop1_hr = np.loadtxt('dl_prop1_hr.dat')
dl_prop1_hi = np.loadtxt('dl_prop1_hi.dat')
diff_prop1_hr = prop1_hr-dl_prop1_hr
diff_prop1_hi = prop1_hi-dl_prop1_hi
print("prop1_hr 的差为：")
print(np.linalg.norm(diff_prop1_hr,ord=2) )
print("prop1_hi 的差为：")
print(np.linalg.norm(diff_prop1_hi,ord=2) )

plt.figure(1, dpi = 300)
plt.plot(prop1_hr, color = "red", label = "prop1_hr")
plt.plot(dl_prop1_hr, color = "blue", label = "dl_prop1_hr")
plt.legend()
plt.savefig("prop1_hr.png")
plt.close()

plt.figure(1, dpi = 300)
plt.plot(prop1_hi, color = "red", label = "prop1_hi")
plt.plot(dl_prop1_hi, color = "blue", label = "dl_prop1_hi")
plt.legend()
plt.savefig("prop1_hi.png")
plt.close()
'''




evol1 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/evol1_inIntensity.dat')
evol1_ur = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/evol1_ur.dat')
evol1_ui = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/evol1_ui.dat')
evol1_ur11 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/evol1_ur11.dat')
evol1_ui11 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/evol1_ui11.dat')
dl_evol1 = np.loadtxt('dl_evol1.dat')
dl_evol1_ur = np.loadtxt('dl_evol1_ur.dat')
dl_evol1_ui = np.loadtxt('dl_evol1_ui.dat')
dl_evol1_ur11 = np.loadtxt('dl_evol1_ur11.dat')
dl_evol1_ui11 = np.loadtxt('dl_evol1_ui11.dat')
diff_evol1 = evol1-dl_evol1
diff_evol1_ur = evol1_ur-dl_evol1_ur
diff_evol1_ui = evol1_ui-dl_evol1_ui
diff_evol1_ur11 = evol1_ur11-dl_evol1_ur11
diff_evol1_ui11 = evol1_ui11-dl_evol1_ui11
print("evol1 的差为：")
print(np.linalg.norm(diff_evol1,ord=2) )
print("evol1_ur 的差为：")
print(np.linalg.norm(diff_evol1_ur,ord=2) )
print("evol1_ui 的差为：")
print(np.linalg.norm(diff_evol1_ui,ord=2) )
print("evol1_ur11 的差为：")
print(np.linalg.norm(diff_evol1_ur11,ord=2) )
print("evol1_ui11 的差为：")
print(np.linalg.norm(diff_evol1_ui11,ord=2) )


plt.figure(1, dpi = 300)
plt.contourf(evol1_ur)
plt.savefig("evol1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_evol1_ur)
plt.savefig("dl_evol1_ur.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(evol1_ui)
plt.savefig("evol1_ui.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_evol1_ui)
plt.savefig("dl_evol1_ui.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(evol1_ur11)
plt.savefig("evol1_ur11.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_evol1_ur11)
plt.savefig("dl_evol1_ur11.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(evol1_ui11)
plt.savefig("evol1_ui11.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_evol1_ui11)
plt.savefig("dl_evol1_ui11.png")
plt.close()



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

plt.figure(1, dpi = 300)
plt.contourf(outIntensity)
plt.savefig("outIntensity.png")
plt.close()

plt.figure(1, dpi = 300)
plt.contourf(dl_outIntensity)
plt.savefig("dl_outIntensity.png")
plt.close()




