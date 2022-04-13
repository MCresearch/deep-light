# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


fft_in_temp_xr1 = np.loadtxt('fft_in_temp_xr1.dat')
dl_fft_in_temp_xr1 = np.loadtxt('dl_fft_in_temp_xr1.dat')
diff_fft_in_temp_xr1 = fft_in_temp_xr1-dl_fft_in_temp_xr1
print("fft_in_temp_xr1 的差为：")
np.savetxt('221.txt', diff_fft_in_temp_xr1,delimiter=',')  
print(np.std(diff_fft_in_temp_xr1) )

fft_in_temp_xr10 = np.loadtxt('fft_in_temp_xr10.dat')
dl_fft_in_temp_xr10 = np.loadtxt('dl_fft_in_temp_xr10.dat')
diff_fft_in_temp_xr10 = fft_in_temp_xr10-dl_fft_in_temp_xr10
print("fft_in_temp_xr10 的差为：") 
print(np.std(diff_fft_in_temp_xr10) )

dl_mdfph1_ur = np.loadtxt('dl_mdfph1_ur.dat')
dl_fft_in_temp_xr1 = np.loadtxt('dl_fft_in_temp_xr1.dat')
diff_dl_mdfph1_ur = dl_mdfph1_ur-dl_fft_in_temp_xr1
print("dl_mdfph1_ur 的差为：")
print(np.linalg.norm(diff_dl_mdfph1_ur,ord=2) )

mdfph1_ur = np.loadtxt('mdfph1_ur.dat')
fft_in_temp_xr10 = np.loadtxt('fft_in_temp_xr10.dat')
diff_mdfph1_ur = mdfph1_ur-fft_in_temp_xr10
print("mdfph1_ur fortran 的差为：")
print(np.linalg.norm(diff_mdfph1_ur,ord=2) )

mdfph1_ur = np.loadtxt('mdfph1_ur.dat')
fft_in_temp_xr100 = np.loadtxt('fft_in_temp_xr100.dat')
diff_mdfph1_ur1 = mdfph1_ur-fft_in_temp_xr100
print("mdfph1_ur fortran 0的差为：")
print(np.linalg.norm(diff_mdfph1_ur1,ord=2) )


mdfph1_ur = np.loadtxt('mdfph1_ur.dat')
dl_mdfph1_ur= np.loadtxt('dl_mdfph1_ur.dat')
diff_dl_mdfph1_ur = dl_mdfph1_ur-mdfph1_ur
print("mdfph1_ur 的差为：")
print(np.linalg.norm(diff_dl_mdfph1_ur,ord=2) )


