# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# dl_inIntensity = np.loadtxt('dl_inIntensity.dat',comments='#')

# plt.figure(1, dpi = 300)
# plt.contourf(dl_inIntensity)
# plt.savefig("./pictures/dl_inIntensity.png")
# plt.close()


# dl_inPhase_all = np.loadtxt('dl_inPhase_1.239100.dat',comments='#')
# dl_inPhase = dl_inPhase_all[0:32,]


# plt.figure(1, dpi = 300)
# plt.contourf(dl_inPhase)
# plt.savefig("./pictures/dl_inPhase.png")
# plt.close()

# dl_focusing_all = np.loadtxt('dl_focusing_1.239100.dat',comments='#')
# dl_focusing = dl_focusing_all[0:32,]
# dl_focusing_ur = dl_focusing_all[32:64,]
# dl_focusing_ui = dl_focusing_all[64:96,]


# plt.figure(1, dpi = 300)
# plt.contourf(dl_focusing)
# plt.savefig("./pictures/dl_focusing.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_focusing_ur)
# plt.savefig("./pictures/dl_focusing_ur.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_focusing_ui)
# plt.savefig("./pictures/dl_focusing_ui.png")
# plt.close()


# dl_mdfph1_all = np.loadtxt('dl_mdfph1_1.239100.dat',comments='#')
# dl_mdfph1 = dl_mdfph1_all[0:32,]
# dl_mdfph1_ur = dl_mdfph1_all[32:64,]
# dl_mdfph1_ui = dl_mdfph1_all[64:96,]


# plt.figure(1, dpi = 300)
# plt.contourf(dl_mdfph1)
# plt.savefig("./pictures/dl_mdfph1.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_mdfph1_ur)
# plt.savefig("./pictures/dl_mdfph1_ur.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_mdfph1_ui)
# plt.savefig("./pictures/dl_mdfph1_ui.png")
# plt.close()



# dl_my_fft2d1_all = np.loadtxt('dl_my_fft2d1_1.239100.dat',comments='#')
# dl_my_fft2d1 = dl_my_fft2d1_all[0:32,]
# dl_my_fft2d1_ur = dl_my_fft2d1_all[32:64,]
# dl_my_fft2d1_ui = dl_my_fft2d1_all[64:96,]



# plt.figure(1, dpi = 300)
# plt.contourf(dl_my_fft2d1)
# plt.savefig("./pictures/dl_my_fft2d1.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_my_fft2d1_ur)
# plt.savefig("./pictures/dl_my_fft2d1_ur.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_my_fft2d1_ui)
# plt.savefig("./pictures/dl_my_fft2d1_ui.png")
# plt.close()


# dl_evol1_all = np.loadtxt('dl_evol1_1.239100.dat',comments='#')
# dl_evol1 = dl_evol1_all[0:32,]
# dl_evol1_ur = dl_evol1_all[32:64,]
# dl_evol1_ui = dl_evol1_all[64:96,]



# plt.figure(1, dpi = 300)
# plt.contourf(dl_evol1)
# plt.savefig("./pictures/dl_evol1.png")
# plt.close()


# plt.figure(1, dpi = 300)
# plt.contourf(dl_evol1_ur)
# plt.savefig("./pictures/dl_evol1_ur.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_evol1_ui)
# plt.savefig("./pictures/dl_evol1_ui.png")
# plt.close()



# dl_my_fft2d2_all = np.loadtxt('dl_my_fft2d2_1.239100.dat',comments='#')
# dl_my_fft2d2 = dl_my_fft2d1_all[0:32,]
# dl_my_fft2d2_ur = dl_my_fft2d2_all[32:64,]
# dl_my_fft2d2_ui = dl_my_fft2d2_all[64:96,]


# plt.figure(1, dpi = 300)
# plt.contourf(dl_my_fft2d2)
# plt.savefig("./pictures/dl_my_fft2d2.png")
# plt.close()


# plt.figure(1, dpi = 300)
# plt.contourf(dl_my_fft2d2_ur)
# plt.savefig("./pictures/dl_my_fft2d2_ur.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_my_fft2d2_ui)
# plt.savefig("./pictures/dl_my_fft2d2_ui.png")
# plt.close()

# dl_mdfph2_all = np.loadtxt('dl_mdfph2_1.239100.dat',comments='#')
# dl_mdfph2 = dl_mdfph1_all[0:32,]
# dl_mdfph2_ur = dl_mdfph1_all[32:64,]
# dl_mdfph2_ui = dl_mdfph1_all[64:96,]


# plt.figure(1, dpi = 300)
# plt.contourf(dl_mdfph2)
# plt.savefig("./pictures/dl_mdfph2.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_mdfph2_ur)
# plt.savefig("./pictures/dl_mdfph2_ur.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_mdfph2_ui)
# plt.savefig("./pictures/dl_mdfph2_ui.png")
# plt.close()

'''
dl_inIntensity_all = np.loadtxt('./zernike_order/dl_inIntensity_104_10000.dat',comments='#')
dl_inIntensity = dl_inIntensity_all[0:64,]
plt.figure(1, dpi = 200)
plt.contourf(dl_inIntensity)
plt.savefig("./dl_inIntensity_no_1000.png")
plt.close()
'''

# dl_outIntensity_all = np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_outIntensity.dat',comments='#')
# a = 4*64
# b = 5*64
# dl_outIntensity = dl_outIntensity_all[a:b,]
# plt.figure(1, dpi = 300)
# plt.contourf(dl_outIntensity)
# plt.savefig("./outIntensity_1_1.png")
# plt.close()
'''
dl_outIntensity_all = np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_outIntensity.dat',comments='#')
plt.figure(1, dpi = 300)
plt.contourf(dl_outIntensity_all)
plt.savefig("./dl_outIntensity.png")
plt.close()

dl_inPhase_all = np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_ph_.dat',comments='#')
print("dl_inPhase_all shape = ", np.shape(dl_inPhase_all))
plt.figure(1, dpi = 300)
plt.contourf(dl_inPhase_all)
plt.savefig("./dl_inPhase_no_10000.png")
plt.close()
'''
# inIntensity_ur = np.loadtxt('/home/xianyuer/yuer/origin_num/inIntensity_ur_256.dat',comments='#')
# inIntensity_ui = np.loadtxt('/home/xianyuer/yuer/origin_num/inIntensity_ui_256.dat',comments='#')
# dl_inIntensity = np.loadtxt('./dl_inIntensity.dat',comments='#')
# dl_inIntensity_ur = dl_inIntensity[64:128,]
# dl_inIntensity_ui = dl_inIntensity[128:192,]
# diff_inIntensity_ur = inIntensity_ur-dl_inIntensity_ur
# diff_inIntensity_ui = inIntensity_ui-dl_inIntensity_ui
# print("inIntensity 的差为：")
# print(np.linalg.norm(diff_inIntensity_ur,ord=2) )
# print(np.linalg.norm(diff_inIntensity_ui,ord=2) )

# plt.figure(1, dpi = 300)
# plt.contourf(inIntensity_ur)
# plt.savefig("inIntensity_ur.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_inIntensity_ur)
# plt.savefig("dl_inIntensity_ur.png")
# plt.close()

# ph = np.loadtxt('/home/xianyuer/yuer/origin_num/inPhase.dat',comments='#')
# dl_ph= np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_ph_.dat',comments='#')
# diff_ph = ph-dl_ph
# print("ph 的差为：")
# print(np.linalg.norm(diff_ph,ord=2) )

# inphase_ur = np.loadtxt('/home/xianyuer/yuer/origin_num/inPhase_ur.dat',comments='#')
# inphase_ui = np.loadtxt('/home/xianyuer/yuer/origin_num/inPhase_ui.dat',comments='#')
# dl_inphase = np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_inPhase_0.131538.dat',comments='#')
# dl_inphase_ur = dl_inphase[64:128,]
# dl_inphase_ui = dl_inphase[128:192,]
# diff_inphase_ur = inphase_ur-dl_inphase_ur
# diff_inphase_ui = inphase_ui-dl_inphase_ui
# print("ur,ui的差为：")
# print(np.linalg.norm(diff_inphase_ur,ord=2) )
# print(np.linalg.norm(diff_inphase_ui,ord=2) )

# plt.figure(1, dpi = 300)
# plt.contourf(dl_inphase_ur)
# plt.savefig("dl_inphase_ur.png")
# plt.close()
# plt.figure(1, dpi = 300)
# plt.contourf(inphase_ur)
# plt.savefig("dl_inphase_ur_0.png")
# plt.close()

# focusing = np.loadtxt('/home/xianyuer/yuer/origin_num/focusing_inIntensity.dat',comments='#')
# dl_focusing = np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_focusing_0.131538.dat',comments='#')
# dl_focusing = dl_focusing[0:64,]
# diff_focusing = focusing-dl_focusing
# print("focusing 的差为：")
# print(np.linalg.norm(diff_focusing,ord=2) )

# mdfph1 = np.loadtxt('/home/xianyuer/yuer/origin_num/mdfph1_inIntensity.dat',comments='#')
# dl_mdfph1 = np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_mdfph1_0.131538.dat',comments='#')
# dl_mdfph1= dl_mdfph1[0:64,]
# diff_mdfph1 = mdfph1-dl_mdfph1
# print("mdfph1 的差为：")
# print(np.linalg.norm(diff_mdfph1,ord=2) )

# fft1 = np.loadtxt('/home/xianyuer/yuer/origin_num/my_fft2d1_inIntensity.dat',comments='#')
# dl_fft1 = np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_my_fft2d1_0.131538.dat',comments='#')
# dl_fft1= dl_fft1[0:64,]
# diff_fft1 = fft1-dl_fft1
# print("fft1 的差为：")
# print(np.linalg.norm(diff_fft1,ord=2) )

# evol = np.loadtxt('/home/xianyuer/yuer/origin_num/evol1_inIntensity.dat',comments='#')
# dl_evol= np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_evol1_0.131538.dat',comments='#')
# dl_evol= dl_evol[0:64,]
# diff_evol = evol-dl_evol
# print("evol1 的差为：")
# print(np.linalg.norm(diff_fft1,ord=2) )

# fft2 = np.loadtxt('/home/xianyuer/yuer/origin_num/my_fft2d2_inIntensity.dat',comments='#')
# dl_fft2 = np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_my_fft2d2_0.131538.dat',comments='#')
# dl_fft2= dl_fft2[0:64,]
# diff_fft2 = fft2-dl_fft2
# print("fft2 的差为：")
# print(np.linalg.norm(diff_fft2,ord=2) )

# mdfph2 = np.loadtxt('/home/xianyuer/yuer/origin_num/mdfph2_inIntensity.dat',comments='#')
# mdfph2_ur = np.loadtxt('/home/xianyuer/yuer/origin_num/mdfph2_ur.dat',comments='#')
# mdfph2_ui = np.loadtxt('/home/xianyuer/yuer/origin_num/mdfph2_ur.dat',comments='#')
# dl_mdfph2 = np.loadtxt('/home/xianyuer/yuer/num_mechinelearning/deep-light/tests/64_64/dl_mdfph2_0.131538.dat',comments='#')
# dl_mdfph2_ur = dl_mdfph2[64:128,]
# dl_mdfph2_ui = dl_mdfph2[128:192,]
# dl_mdfph2= dl_mdfph2[0:64,]


# diff_mdfph2 = mdfph2-dl_mdfph2
# diff_mdfph2_ur = mdfph2_ur-dl_mdfph2_ur
# diff_mdfph2_ui = mdfph2_ui-dl_mdfph2_ui
# print("mdfph2 的差为：")
# print(np.linalg.norm(diff_mdfph2_ur,ord=2) )
# print(np.linalg.norm(diff_mdfph2_ui,ord=2) )
# print(np.linalg.norm(diff_mdfph2,ord=2) )
# plt.figure(1, dpi = 300)
# plt.contourf(mdfph2_ui)
# plt.savefig("dl_mdfph2_ui_0.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_mdfph2_ui)
# plt.savefig("dl_mdfph2_ui_.png")
# plt.close()

# plt.contourf(mdfph2_ur)
# plt.savefig("dl_mdfph2_ur_0.png")
# plt.close()

# plt.figure(1, dpi = 300)
# plt.contourf(dl_mdfph2_ur)
# plt.savefig("dl_mdfph2_ur_.png")
# plt.close()


outIntensity = np.loadtxt('./fortran/outIntensity.dat',comments='#')
dl_outIntensity = np.loadtxt('./dl_outIntensity.dat',comments='#')
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






