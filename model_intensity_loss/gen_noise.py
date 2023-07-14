import numpy as np
import os
import matplotlib.pyplot as plt
import math
rms = 4
nsnapshots = 1000
n_grid = 128
interval = 5
aaz = 0.2196/2
dir = "/home/xianyuer/data/intensityloss_git/zer9/5_"
for k in range(1):
    intensity = np.load("/home/xianyuer/data/intensityloss_git/zer9/5_1000_sumnor_outintensity_y3.npy")
    D = 0.0001
    x = intensity.copy()
    x1 = np.zeros((nsnapshots,128,128))
    print(np.shape(intensity))
    noise = np.zeros((nsnapshots,n_grid,n_grid))
    e_noise = np.zeros((nsnapshots))
    f_int = np.zeros((nsnapshots))
    SNR = np.zeros((nsnapshots))
    SNR_db = np.zeros((nsnapshots))
    for j in range(nsnapshots):
        noise[j,:,:] = np.random.normal(0,D,(n_grid,n_grid))
        # print(noise)
        x1[j,:,:] = x[j,:,:] + noise[j,:,:]
        for ii in range(n_grid):
            for jj in range(n_grid):
                if x1[j,ii,jj] < 0:
                    x1[j,ii,jj] = 0
                if x1[j,ii,jj] > 1:
                    x1[j,ii,jj] = 1
    print(x1)
    # np.save(dir+"%d_2000_noise_norout_%.4lf.npy"%(k,D),x1)
    # nor_outIntensity = np.zeros((nsnapshots,128,128))
    # for j in range(nsnapshots):
    #     nor_outIntensity[j,:,:] = x1[j,:,:]/np.max(x1[j,:,:]) 
    np.save(dir+"new_1000_noise_out_%.4lf.npy"%(D),x1)
    fid = open(dir+"SNR"+"new_1000_noise_out_%.4lf"%(D)+'.log', 'w')
    for i in range(nsnapshots):
        e_noise[i] = np.sum(pow(noise[i,:,:],2))
        f_int[i] = np.sum(pow(x1[i,:,:],2))
        SNR[i] = f_int[i]/e_noise[i]
        SNR_db[i] = 10*math.log10(SNR[i])
        fid.write(str(SNR[i])+'\t'+str(SNR_db[i]))
        fid.write('\n')
    # print(np.max(x))
    # MSE = np.mean((nor_outIntensity-x)*(nor_outIntensity-x))
    # psnr = 10*math.log10(1/MSE)
    # print(MSE)
    print("mean SNR_db",np.mean(SNR_db))
    
    plt.figure(1, dpi=400)
    plt.contourf(x1[0,:,:],levels=100, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/ 2) for i in range(-2,3)],size=10)
    plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/2) for i in range(-2,3)],size=10)
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)
    plt.savefig((dir+"new_1000_noise_out_%.4lf.png")%(D),bbox_inches='tight')
    plt.close()
    # print(np.max(x))
    # MSE = np.mean((nor_outIntensity-x)*(nor_outIntensity-x))
    # psnr = 10*math.log10(1/MSE)
    # print(MSE)
    # print(psnr)
    # noise_2 = np.linalg.norm(noise,ord=2,axis=None)
    # x1_2 = np.linalg.norm(x1,ord=2,axis=None)
    # SNR = np.max(x1)/D
    # SNR_db = 20*math.log(SNR)
    # print(SNR," ",SNR_db)