import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nzernike = 9
framnum =0 #0-10
nsnapshots = 1000
n_grid = 256
m_grid = 128
rms = 4
aaz = 0.2196/2
aa0 = 1.2
interval = 5
dir = "/home/xianyuer/data/intensityloss_git/test_model/0711_9y3_128_intloss_intmean_rms4_200000_b16_e50__step_50_lr_0.0001/"
# dir = "/home/xianyuer/data/intensityloss_git/test_model/0711_9y3_128_int1e7+zerloss50_intmean_rms4_200000_b16_e50_step_50_lr_0.0001/noise0.0001/"
diff_out_nor = np.load(dir+"diff_sumnor_outintensity.npy")
# real_out_all = np.load("/home/xianyuer/data/intensityloss_git/zer9/5_new_1000_noise_out_0.0001.npy")
real_out_all = np.load("/home/xianyuer/data/intensityloss_git/zer9/5_1000_sumnor_outintensity_y3.npy")

real_zernike_dir = dir+"zernike_test_real.txt"
predict_zernike_dir = dir+"zernike_test_predict.txt"
diff_1 = np.loadtxt(dir+"zernike_test_diff.txt")
##############zernike  predict corfficient and real corfficient#############
real_zernike = np.loadtxt(real_zernike_dir)
predict_zernike = np.loadtxt(predict_zernike_dir)
x = np.zeros((nzernike-2))
for i in range(nzernike-2):
    x[i] = 3+i

plt.figure(1, dpi = 300)
plt.rc('axes', axisbelow=True)
plt.grid(linestyle='-.',alpha=0.4)
plt.bar(x,real_zernike[framnum,:], color="dodgerblue",alpha=1,label = "Initial values")
plt.bar(x,predict_zernike[framnum,:], color="darkgreen",alpha=0.6,label = "Predict (by Xception)")
plt.xlabel("Zernike order",fontsize=15)
plt.ylabel("Zernike coefficients",fontsize=15)
# print(np.array(range(2,35,5)))
plt.xticks([i+3 for i in range(0,nzernike-2,2)],size = 10)
plt.yticks(size=10)
plt.legend(loc = 'upper right')
plt.savefig(dir+str(framnum+1)+"_zernike.png",bbox_inches='tight') #,bbox_inches='tight'
plt.close()

###################### outintensity fig ############################

plt.figure(1,dpi=300)
plt.contourf(real_out_all[framnum,:,:],levels=100, cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.yticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)
plt.savefig((dir+str(framnum+1)+"_real_int.png"),bbox_inches='tight')
plt.close()

plt.figure(1,dpi=300)
plt.contourf(diff_out_nor[framnum,:,:],levels=100, cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.yticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)
plt.savefig((dir+str(framnum+1)+"_diff_int.png"),bbox_inches='tight')
plt.close()


rms_diff = np.zeros(nsnapshots)
for i in range(nsnapshots):
    for j in range(nzernike-2):
        rms_diff[i] = rms_diff[i]+pow(diff_1[i,j],2)
        
plt.figure(1,dpi=300)
plt.contourf(diff_out_nor[framnum,:,:],levels=100, cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.yticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
plt.xlabel("x (m)",fontsize=15)
plt.ylabel("y (m)",fontsize=15)
plt.savefig((dir+str(framnum+1)+"_diff_int.png"),bbox_inches='tight')
plt.close()

np.savetxt(dir+"rms.txt", np.sqrt(rms_diff/35))
print("rms",np.mean(np.sqrt(rms_diff/35)))
# ######################### SR ######################################33333
rms_real = np.zeros(nsnapshots)
for i in range(nsnapshots):
    rms_real[i] = rms
x4 = np.hstack([np.array(range(0,nsnapshots,200)),np.array(range(0,nsnapshots,200))])
x2 = np.array(range(nsnapshots))
x3 = np.array(range(nsnapshots,2*nsnapshots))
plt.figure(2,dpi = 300)
plt.plot(x2,rms_real,label="real",color="blue")
plt.plot(x3,rms_diff, label="predict",color="red")
plt.xticks([i for i in range (0,2*nsnapshots,200)],x4,size=10)
plt.yticks([i for i in range(5)],size=10)
plt.xlabel("Frame M",fontsize=15)
plt.ylabel("RMS(/rad2)",fontsize=15)
plt.rcParams.update({'font.size':13})
plt.legend()
plt.savefig((dir+"rms.png"),bbox_inches='tight')
plt.close() 
