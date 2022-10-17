import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nzernike = 35
option_zernike_corfficient = 1
option_rms = 1
option_intensity = 1
option_SR = 0
framnum = 10 #0-10
nsnapshots = 1000
distribution = "0-1"
n_grid = 256
m_grid = 128
rms = np.pi
aaz = 0.0061*20
aa0 = 1.2
interval = 5

dir = "/home/xianyuer/data/35_xception_200000_rms1/test/"

##############zernike  predict corfficient and real corfficient#############
if option_zernike_corfficient == 1:
    
    real_zernike_dir = dir+"zernike_test_real.txt"
    predict_zernike_dir = dir+"zernike_test_predict.txt"

    real_zernike = np.loadtxt(real_zernike_dir)
    predict_zernike = np.loadtxt(predict_zernike_dir)

    plt.figure(1, dpi = 400)
    plt.bar(np.array(range(33)),real_zernike[framnum], color="red",alpha=1,label = "real")
    plt.bar(np.array(range(33)),predict_zernike[framnum], color="blue",alpha=0.5,label = "predict")
    plt.xlabel("Zernike order",fontsize=15)
    plt.ylabel("Zernike coefficient values",fontsize=15)
    plt.xticks([i for i in range(33)],[(i+3) for i in range(33)],size = 5)
    plt.yticks(size=10)
    plt.title("Test set No.%d, model = 35_Xceptionm4_pi" % 2,fontsize=15)
    plt.legend()
    plt.savefig(dir+str(framnum)+"_test_xceptionm4_35_1.png",bbox_inches='tight') #,bbox_inches='tight'
    plt.close()

######################   rms   #################################
if option_rms == 1:
    diff_zernike = np.loadtxt(dir+"zernike_test_diff.txt")
    rms_real = np.zeros(nsnapshots)
    rms_diff = np.zeros(nsnapshots)
    for i in range(nsnapshots):
        rms_real[i] = rms
        for j in range(nzernike-2):
            rms_diff[i] = rms_diff[i]+pow(diff_zernike[i,j],2)

    x4 = np.hstack([np.array(range(0,nsnapshots,200)),np.array(range(0,nsnapshots,200))])
    x2 = np.array(range(nsnapshots))
    x3 = np.array(range(nsnapshots,2*nsnapshots))
    plt.figure(2,dpi = 300)
    plt.plot(x2,rms_real,label="real",color="blue")
    plt.plot(x3,rms_diff, label="predict",color="red")
    plt.xticks([i for i in range (0,2*nsnapshots,200)],x4,size=10)
    plt.yticks([i for i in range(5)],size=10)
    plt.xlabel("Frame M",fontsize=15)
    plt.ylabel("RMS2(/rad2)",fontsize=15)
    plt.rcParams.update({'font.size':13})
    plt.legend()
    plt.savefig((dir+"rms.png"),bbox_inches='tight')
    plt.close() 
    print("rms",np.mean(np.sqrt(rms_diff)))
rmsone = []
rmstwo = []
for i in range(nsnapshots):
    if(rms_diff[i]>1) :
        rmsone.append(i)
    if(rms_diff[i]>2) :
        rmstwo.append(i)
np.savetxt(dir+"rms_1.txt",rmsone)
np.savetxt(dir+"rms_2.txt",rmstwo)
print("min",np.min(rms_diff))

###################### outintensity fig ############################
if option_intensity == 1:
    
    real_out_all_0 = np.loadtxt("/home/xianyuer/data/35_xception_200000_rms1/data/5_dl_down_intensity.dat")
    real_out_all = np.zeros((nsnapshots,128,128))
    for i in range(nsnapshots):
        real_out_all[i,:,:] = real_out_all_0[128*i:128*(i+1),:]
    diff_out_all = np.load("/home/xianyuer/data/35_xception_200000_rms1/test/outintensity.npy")

    plt.figure(1, dpi=400)
    plt.contourf(real_out_all[framnum,:,:],levels=[i*interval for i in range(int(np.max(real_out_all[framnum,:,:])/interval + 2*interval))], cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xticks([i*m_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.yticks([i*m_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)
    plt.savefig((dir+str(framnum)+"outIntensity_real.png"),bbox_inches='tight')
    plt.close()

    plt.figure(1, dpi=400)
    plt.contourf(diff_out_all[framnum,:,:],levels=[i*interval for i in range(int(np.max(diff_out_all[framnum,:,:])/interval + 2*interval))], cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xticks([i*m_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.yticks([i*m_grid/4 for i in range(5)], ["%.4f" %(i*aaz/4) for i in range(-2,3)],size=10)
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)
    plt.savefig((dir+str(framnum)+"outIntensity_diff.png"),bbox_inches='tight')
    plt.close()

######################### SR ######################################33333
# if option_SR == 1:
ideal_out = np.loadtxt("/home/xianyuer/data/35_xception_200000_rms1/test/ideal_dl_down_intensity.dat")
ideal = np.max(ideal_out)
max_real = 0
max_diff = 0
for i in range(nsnapshots):
    max_diff = np.max(diff_out_all[i,:,:])+max_diff
    max_real = np.max(real_out_all[i,:,:])+max_real

print("SR_diff = ",max_diff/nsnapshots/ideal)
print("SR_real = ",max_real/nsnapshots/ideal)

'''
###################################p########################
length = 256

def get_mass_center(value):
    x_moment = 0
    y_moment = 0
    for iy in range(length):
        for ix in range(length):
            intens = value[iy][ix]
            x_moment += ix*intens
            y_moment += iy*intens
    intens_total = np.sum(value)
    x_center = x_moment / intens_total
    y_center = y_moment / intens_total

    return (y_center, x_center)

def get_energy_r(value, mass_center):
    energy_profile = np.zeros(length*2)
    nr = np.zeros(length*2)
    for iy in range(length):
        for ix in range(length):
            xcoord = int(np.abs(ix - mass_center[1]))
            ycoord = int(np.abs(iy - mass_center[0]))
            r = int(np.sqrt(xcoord**2 + ycoord**2))
            energy_profile[r] += value[iy][ix]
            nr[r] += 1
    for ir in range(length*2):
        if nr[ir] > 0:
            energy_profile[ir] /= nr[ir]
    return energy_profile

def get_rcut(energy_profile, percentage):
    energy_int = []
    rcut = 0
    intens_cut = 0
    energy_sum = 2*np.pi*np.sum(np.array([i*energy_profile[i] for i in range(2*length)]))
    sum_rcut = 0
    for ir in range(2*length):
        sum_rcut += 2*np.pi*ir*energy_profile[ir]
        if sum_rcut/energy_sum > percentage and intens_cut == 0:
            rcut = ir
            intens_cut = energy_profile[ir]
        energy_int.append(sum_rcut/energy_sum)
    return rcut, intens_cut, energy_int


real = diff_out
# real = real_out_all
value = np.zeros((nsnapshots,length,length))
r = np.zeros(nsnapshots)
p = np.zeros(nsnapshots)
# real = np.genfromtxt(dir+"dl_outIntensity.dat")
#print(get_mass_center(real[:length, ]))
#print(get_mass_center(real[length:2*length, ]))
for i in range(nsnapshots):
    value[i,:,:] = real[i,:length, ]/np.max(real[i,:length, ])
    mass_center = get_mass_center(value[i,:,:])
    # print("mass_center",mass_center)
    energy_profile = get_energy_r(value[i,:,:], mass_center)
    # print(get_rcut(energy_profile, 0.84))
    rcut, intens_cut, energy_int = get_rcut(energy_profile, 0.84)
    airy = 0.0061
    aaz = 36
    r[i] = rcut*airy*aaz/256
    p[i] = r[i]/airy
np.savetxt(dir+"r.txt",p)
print("p",np.mean(p))

# # plt.figure(1,dpi=300)
# # plt.plot(np.array(range(1,11)),r)
# # plt.xlabel("Frame M")
# # plt.ylabel("Radius(m)")
# # plt.savefig(dir+"radius.png",bbox_inches='tight')
'''