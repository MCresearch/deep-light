import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nzernike = 35
framnum =1 #0-10
nsnapshots = 4000
distribution = "guass"
n_grid = 256
m_grid = 128
rms = 2
aaz = 0.2196/2
#aaz = 0.122/2
aa0 = 1.2
interval = 5
# dir = "/home/xianyuer/data/paper1/model_rmsmix/rmstest_noise0.01/"
dir = "/home/xianyuer/data/yuer/testwej/deep-light/model_intensity_loss/repro/zer35_rms4/sumnor/35_128_precenter36_zernike+intloss_rms4_b16_e500000_lr_0.0001_intloss_rms4_b16_e100000_lr_0.0001/"
real_out_all = np.load("/home/xianyuer/data/yuer/testwej/deep-light/model_intensity_loss/repro/zer35_rms4/sumnor/testdata/rms1234_4000_128_35_gauss.npy")
# real_out_all = np.load("/home/xianyuer/data/yuer/testwej/deep-light/model_intensity_loss/repro/zer35_rms4/sumnor/3dataforluo/3_128_65__sumnor_guass.npy")
intensity_dir = dir+"diff_rms4321_4000_128_35.npy"
diff_out_nor = np.load(intensity_dir)

real_zernike_dir = dir+"zernike_test_real.txt"
predict_zernike_dir = dir+"zernike_test_predict.txt"
diff_1 = np.loadtxt(dir+"zernike_test_diff.txt")
fid = open(dir+'rms.log', 'w')
# fid_sr = open(dir+'SR.log', 'w')
# if os.path.exists(dir):
#     print("已存在")
# else:
#     os.mkdir(dir)
    
##############zernike  predict corfficient and real corfficient#############
real_zernike = np.loadtxt(real_zernike_dir)
predict_zernike = np.loadtxt(predict_zernike_dir)
x = np.zeros((33))
for i in range(33):
    x[i] = 3+i
for i in range(4):
    plt.figure(1, dpi = 600)
    plt.rc('axes', axisbelow=True)
    plt.grid(linestyle='-.',alpha=0.4)
    plt.bar(x,real_zernike[i*1000,:], color="dodgerblue",alpha=1,label = "Initial values")
    plt.bar(x,predict_zernike[i*1000,:], color="darkgreen",alpha=0.6,label = "Predict (by Xception)")
    plt.xlabel("Zernike order",fontsize=15)
    plt.ylabel("Zernike coefficients",fontsize=15)
    # print(np.array(range(2,35,5)))
    plt.xticks([i+3 for i in range(0,33,2)],size = 10)
    plt.yticks(size=10)
    plt.legend(loc = 'upper right')
    plt.savefig(dir+"rms_"+str(i+1)+"_zernike.png",bbox_inches='tight') #,bbox_inches='tight'
    plt.close()

###################### outintensity fig ############################

for i in range(4):
    plt.figure(1,dpi=600)
    plt.contourf(real_out_all[i*1000,:,:],levels=100, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.yticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)
    plt.savefig((dir+"rms_"+str(i+1)+"_real_int.png"),bbox_inches='tight')
    plt.close()
    
    plt.figure(1,dpi=600)
    plt.contourf(diff_out_nor[i*1000,:,:],levels=100, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.yticks([i*m_grid/4 for i in range(5)], ["%.2f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)
    plt.savefig((dir+"rms_"+str(i+1)+"_diff_int.png"),bbox_inches='tight')
    plt.close()


rms_diff = np.zeros(nsnapshots)
rms_diff_1 = np.zeros((1000,4))
rms_diff_2 = np.zeros((1000,4))
for i in range(nsnapshots):
    for j in range(33):
        rms_diff[i] = rms_diff[i]+pow(diff_1[i,j],2)
for i in range(4):
    rms_diff_1[:,i] = rms_diff[i*1000:(i+1)*1000]
    print("rs_mean",str(i+1)+":",np.mean(np.sqrt(rms_diff_1[:,i])))        
    print("rms_mean",str(i+1)+":",np.mean(np.sqrt(rms_diff_1[:,i]/35)))
    fid.write(str(np.mean(np.sqrt(rms_diff_1[:,i])))+'\t'+str(np.mean(np.sqrt(rms_diff_1[:,i]/35))))
    fid.write('\n')
    rms_diff_2[:,i] = np.sqrt(rms_diff_1[:,i]/35)
np.savetxt(dir+"rms.txt", rms_diff_2)

# ######################### SR ######################################33333
# ideal = 1187.739988
# max_real = np.zeros((4))
# max_diff = np.zeros((4))
# SR_real = np.zeros((4))
# SR_diff = np.zeros((4))
# for i in range(4):
#     for j in range(1000):
#         max_diff[i] = np.max(diff_out[i*1000+j,:,:])+max_diff[i]
#         # max_real[i] = np.max(real_out[i*1000+j,:,:])+max_real[i]
#     SR_diff[i] =max_diff[i]/1000/ideal
#     # SR_real[i] =max_real[i]/1000/ideal
#     fid_sr.write(str(SR_diff[i]))
#     fid_sr.write("\n")