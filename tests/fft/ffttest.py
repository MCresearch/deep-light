# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fft_initialize(mm, n, fft)
'''
fft_in_wr = np.loadtxt('fft_in_wr.dat')
dl_fft_in_wr = np.loadtxt('dl_fft_in_wr.dat')
diff_fft_in_wr = fft_in_wr-dl_fft_in_wr
print("fft_in_wr 的差为：")
print(np.linalg.norm(diff_fft_in_wr,ord=2) )

plt.figure(1, dpi = 300)
plt.plot(fft_in_wr,color="blue",label = "fft_in_wr")
plt.plot(dl_fft_in_wr,color="red",label = "dl_fft_in_wr")
plt.legend()
plt.savefig("fft_in_wr.png")
plt.close()


fft_in_wi = np.loadtxt('fft_in_wi.dat')
dl_fft_in_wi = np.loadtxt('dl_fft_in_wi.dat')
diff_fft_in_wi = fft_in_wi-dl_fft_in_wi
print("fft_in_wi 的差为：")
print(np.linalg.norm(diff_fft_in_wi,ord=2) )

plt.figure(1, dpi = 300)
plt.plot(fft_in_wi,color="blue",label = "fft_in_wi")
plt.plot(dl_fft_in_wi,color="red",label = "dl_fft_in_wi")
plt.legend()
plt.savefig("fft_in_wi.png")
plt.close()

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
'''

# fft2d
'''
fft_in_temp_xr1 = np.loadtxt(
    '/home/xianyuer/yuer/numerical_diffraction/mohan_tests/fft_in_temp_xr1.dat')
dl_fft_in_temp_xr1 = np.loadtxt(
    'dl_fft_in_temp_xr1.dat')
diff_fft_in_temp_xr1 = fft_in_temp_xr1-dl_fft_in_temp_xr1
print("fft_in_temp_xr1 的差为：")
np.savetxt('fft_in_temp_xr1.txt', diff_fft_in_temp_xr1, delimiter=',')
print(np.std(diff_fft_in_temp_xr1))

plt.figure(1, dpi=300)
plt.contourf(fft_in_temp_xr1)
plt.savefig("fft_in_temp_xr1.png")
plt.close()

plt.figure(1, dpi=300)
plt.contourf(dl_fft_in_temp_xr1)
plt.savefig("dl_fft_in_temp_xr1.png")
plt.close()


fft_in_temp_xi1 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/fft_in_temp_xi1.dat')
dl_fft_in_temp_xi1 = np.loadtxt('dl_fft_in_temp_xi1.dat')
diff_fft_in_temp_xi1 = fft_in_temp_xi1-dl_fft_in_temp_xi1
print("fft_in_temp_xi1 的差为：")
np.savetxt('fft_in_temp_xi1.txt', diff_fft_in_temp_xi1, delimiter=',')
print(np.std(diff_fft_in_temp_xi1))

plt.figure(1, dpi=300)
plt.contourf(fft_in_temp_xi1)
plt.savefig("fft_in_temp_xi1.png")
plt.close()

plt.figure(1, dpi=300)
plt.contourf(dl_fft_in_temp_xi1)
plt.savefig("dl_fft_in_temp_xi1.png")
plt.close()
'''

fft_in_temp_cr1 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/fft_in_temp_cr1.dat')
fft_in_temp_cr1 = np.delete(
    fft_in_temp_cr1, [256, 257, 258, 259, 260, 261, 262, 263, 264], axis=1)
fft_in_temp_cr1 = np.delete(
    fft_in_temp_cr1, [256, 257, 258, 259, 260, 261, 262, 263, 264], axis=0)
dl_fft_in_temp_cr1 = np.loadtxt(
    'dl_fft_in_temp_cr1.dat')
dl_fft_in_temp_cr1 = dl_fft_in_temp_cr1[256:]
diff_fft_in_temp_cr1 = fft_in_temp_cr1-dl_fft_in_temp_cr1
print("fft_in_temp_cr1 的差为：")
np.savetxt('fft_in_temp_cr1.txt', diff_fft_in_temp_cr1, delimiter=',')
print(np.std(diff_fft_in_temp_cr1))

plt.figure(1, dpi=300)
plt.contourf(fft_in_temp_cr1)
plt.savefig("fft_in_temp_cr1.png")
plt.close()

plt.figure(1, dpi=300)
plt.contourf(dl_fft_in_temp_cr1)
plt.savefig("dl_fft_in_temp_cr1.png")
plt.close()


fft_in_temp_ci1 = np.loadtxt('/home/xianyuer/yuer/numerical_diffraction/mohan_tests/fft_in_temp_ci1.dat')
fft_in_temp_ci1 = np.delete(
    fft_in_temp_ci1, [256, 257, 258, 259, 260, 261, 262, 263, 264], axis=1)
fft_in_temp_ci1 = np.delete(
    fft_in_temp_ci1, [256, 257, 258, 259, 260, 261, 262, 263, 264], axis=0)
dl_fft_in_temp_ci1 = np.loadtxt(
    'dl_fft_in_temp_ci1.dat')
dl_fft_in_temp_ci1 = dl_fft_in_temp_ci1[256:]
diff_fft_in_temp_ci1 = fft_in_temp_ci1-dl_fft_in_temp_ci1
print("fft_in_temp_ci1 的差为：")
np.savetxt('fft_in_temp_ci1.txt', diff_fft_in_temp_ci1, delimiter=',')
print(np.std(diff_fft_in_temp_ci1))

plt.figure(1, dpi=300)
plt.contourf(fft_in_temp_ci1)
plt.savefig("fft_in_temp_ci1.png")
plt.close()

plt.figure(1, dpi=300)
plt.contourf(dl_fft_in_temp_ci1)
plt.savefig("dl_fft_in_temp_ci1.png")
plt.close()


