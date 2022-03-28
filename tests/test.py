# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

inIntensity = np.loadtxt('inIntensity.dat')
dl_inIntensity = np.loadtxt('dl_inIntensity.dat')
diff_inIntensity = inIntensity-dl_inIntensity
print("inIntensity 的差为：")
print(np.linalg.norm(diff_inIntensity,ord=2) )


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


inPhase = np.loadtxt('inPhase.dat')
dl_inPhase = np.loadtxt('dl_inPhase.dat')
diff_inPhase = inPhase-dl_inPhase
print("inPhase 的差为：")
print(np.linalg.norm(diff_inPhase,ord=2) )


inPhase_intensity = np.loadtxt('inPhase_intensity.dat')
dl_inPhase_intensity = np.loadtxt('dl_inPhase_intensity.dat')
diff_inPhase_intensity = inPhase_intensity-dl_inPhase_intensity
print("inPhase_intensity 的差为：")
print(np.linalg.norm(diff_inPhase_intensity,ord=2) )

fft_initialize = np.loadtxt('fft_initialize.dat')
dl_fft_initialize = np.loadtxt('dl_fft_initialize.dat')
diff_fft_initialize = fft_initialize-dl_fft_initialize
print("fft_initialize 的差为：")
print(np.linalg.norm(diff_fft_initialize,ord=2) )


focusing = np.loadtxt('focusing.dat')
dl_focusing = np.loadtxt('dl_focusing.dat')
diff_focusing = focusing-dl_focusing
print("focusing 的差为：")
print(np.linalg.norm(diff_focusing,ord=2) )

mdfph1 = np.loadtxt('mdfph1.dat')
dl_mdfph1 = np.loadtxt('dl_mdfph1.dat')
diff_mdfph1 = mdfph1-dl_mdfph1
print("mdfph1 的差为：")
print(np.linalg.norm(diff_mdfph1,ord=2) )


my_fft2d1 = np.loadtxt('my_fft2d1.dat')
dl_my_fft2d1 = np.loadtxt('dl_my_fft2d1.dat')
diff_my_fft2d1 = my_fft2d1 - dl_my_fft2d1
print("my_fft2d1 的差为：")
print(np.linalg.norm(diff_my_fft2d1,ord=2) )
np.max(my_fft2d1)

prop1 = np.loadtxt('prop1.dat')
dl_prop1 = np.loadtxt('dl_prop1.dat')
diff_prop1 = prop1-dl_prop1
print("prop1 的差为：")
print(np.linalg.norm(diff_prop1,ord=2) )

evol1 = np.loadtxt('evol1.dat')
dl_evol1 = np.loadtxt('dl_evol1.dat')
diff_evol1 = evol1-dl_evol1
print("evol1 的差为：")
print(np.linalg.norm(diff_evol1,ord=2) )

my_fft2d2 = np.loadtxt('my_fft2d2.dat')
dl_my_fft2d2= np.loadtxt('dl_my_fft2d2.dat')
diff_my_fft2d2 = my_fft2d2-dl_my_fft2d2
print("my_fft2d2 的差为：")
print(np.linalg.norm(diff_my_fft2d2,ord=2) )

mdfph2 = np.loadtxt('mdfph2.dat')
dl_mdfph2 = np.loadtxt('dl_mdfph2.dat')
diff_mdfph2 = mdfph2-dl_mdfph2
print("mdfph2 的差为：")
print(np.linalg.norm(diff_mdfph2,ord=2) )


outIntensity = np.loadtxt('outIntensity.dat')
dl_outIntensity = np.loadtxt('dl_outIntensity.dat')
diff_outIntensity = outIntensity-dl_outIntensity
print("outIntensity 的差为：")
print(np.linalg.norm(diff_outIntensity,ord=2) )

fft_in_wr = np.loadtxt('fft_in_wr.dat')
dl_fft_in_wr = np.loadtxt('dl_fft_in_wr.dat')
diff_fft_in_wr = fft_in_wr-dl_fft_in_wr
print("fft_in_wr 的差为：")
print(np.linalg.norm(diff_fft_in_wr,ord=2) )

fft_in_wi = np.loadtxt('fft_in_wi.dat')
dl_fft_in_wi = np.loadtxt('dl_fft_in_wi.dat')
diff_fft_in_wi = fft_in_wi-dl_fft_in_wi
print("fft_in_wi 的差为：")
print(np.linalg.norm(diff_fft_in_wi,ord=2) )

fft_in_km_0 = np.loadtxt('fft_in_km_0.dat')
dl_fft_in_km_0 = np.loadtxt('dl_fft_in_km_0.dat')
diff_fft_in_km_0 = fft_in_km_0-dl_fft_in_km_0
print("fft_in_km_0 的差为：")
print(np.linalg.norm(diff_fft_in_km_0,ord=2) )

fft_in_km_1 = np.loadtxt('fft_in_km_1.dat')
dl_fft_in_km_1 = np.loadtxt('dl_fft_in_km_1.dat')
diff_fft_in_km_1 = fft_in_km_1 - dl_fft_in_km_1
print("fft_in_km_1 的差为：")
print(np.linalg.norm(diff_fft_in_km_1,ord=2) )

fft_in_km_2 = np.loadtxt('fft_in_km_2.dat')
dl_fft_in_km_2 = np.loadtxt('dl_fft_in_km_2.dat')
diff_fft_in_km_2 = fft_in_km_2-dl_fft_in_km_2
print("fft_in_km_2 的差为：")
print(np.linalg.norm(diff_fft_in_km_2,ord=2) )

fft_in_kk0 = np.loadtxt('fft_in_kk0.dat')
dl_fft_in_kk0 = np.loadtxt('dl_fft_in_kk0.dat')
diff_fft_in_kk0 = fft_in_kk0-dl_fft_in_kk0
print("fft_in_kk0 的差为：")
print(np.linalg.norm(diff_fft_in_kk0,ord=2) )

fft_in_kj0 = np.loadtxt('fft_in_kj0.dat')
dl_fft_in_kj0 = np.loadtxt('dl_fft_in_kj0.dat')
diff_fft_in_kj0 = fft_in_kj0-dl_fft_in_kj0
print("fft_in_kj0 的差为：")
print(np.linalg.norm(diff_fft_in_kj0,ord=2) )

fft_in_km0 = np.loadtxt('fft_in_km0.dat')
dl_fft_in_km0 = np.loadtxt('dl_fft_in_km0.dat')
diff_fft_in_km0 = fft_in_km0-dl_fft_in_km0
print("fft_in_km0 的差为：")
print(np.linalg.norm(diff_fft_in_km0,ord=2) )



