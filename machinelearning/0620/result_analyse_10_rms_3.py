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
aa0 = 1.2
interval = 5

num = 1
rms = 4
aaz0 = 34.92


# xxz = [34.92, 35.03, 35.14, 35.24, 35.35, 35.46, 35.57,35.68, 35.78, 35.89, 36.00, 36.11, 36.22, 36.32,36.43, 36.54, 36.65, 36.76, 36.86, 36.97, 37.08]
xxz = [-0.5,  -0.4, -0.3, -0.2, 0.2, 0.3, 0.4,
                        0.5]
for ii in range(1,9):
    # dir = "/home/xianyuer/data/65_rms4_aaz36_200000_mid4/rms%d/"%rms
    dir = "/home/xianyuer/data/35_rms4_aaz36_200000/defocus/predict_b/%d"%(ii)
    print(dir)
    # aaz = 0.0061*xxz[ii-1]/2
    aaz = 0.0061*36/2
    # real_out_all = np.load("/home/xianyuer/data/35_rms4_aaz36_200000/5_2000nor_outIntensity_65_0-1_4_2000.npy")
    # real_out_all = np.load("/home/xianyuer/data/35_rms4_aaz36_200000/rms4_aaz/%d_%.2f_nor_outIntensity_35_0-1_%d_1000.npy"%(num,aaz0,rms))
    # real_out_all = np.load("/home/xianyuer/data/65_rms4_aaz36_100000/11_2000_nor_outIntensity_65_0,1_4_128.npy")
    # real_out_all = np.load("/home/xianyuer/data/65_rms4_aaz36_100000/rms%dxxz36_10/1000_nor_outIntensity_65_0-1_%s_128.npy"%(rms,rms))

    # real_out_all_1 = np.loadtxt("/home/xianyuer/data/35_rms4_aaz36_200000/rms4_aaz/%d_%.2f_dl_outIntensity.dat"%(num,aaz0))
    # real_out_all = np.zeros((nsnapshots,256,256))
    # for i in range(nsnapshots):
    #     real_out_all[i,:,:]=real_out_all_1[i*256:(i+1)*256,:]

    real_out_all = np.load("/home/xianyuer/data/35_rms4_aaz36_200000/defocus/b_outIntensity_35.npy")
    diff_out = np.load("/home/xianyuer/data/35_rms4_aaz36_200000/defocus/predict_b/b_diff_outIntensity_35.npy")
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
        print(np.array(range(0,61,5)))
        plt.xticks([i for i in range(33)],[(i+3) for i in range(33)],size = 5)
        plt.yticks(size=10)
        plt.title("Test set No.%d, model = xception_midloop4_35" % 2,fontsize=15)
        plt.legend()
        plt.savefig(dir+str(framnum)+"_test_xception_midloop4_35.png",bbox_inches='tight') #,bbox_inches='tight'
        plt.close()


        # x1 = np.zeros(nzernike-2)
        # for i in range(nzernike-2):
        #     x1[i] = i+3
        # for i in range(nsnapshots):
        #     plt.figure(1, dpi=400,figsize=(20,15))
        #     total_width, n = 0.8, 2
        #     # 每种类型的柱状图宽度
        #     width = total_width / n
        #     plt.bar(x1,real_zernike[i,], width=width, label="real",color="blue")
        #     plt.bar(x1+width,predict_zernike[i,], width=width, label="predict",color="red")
        #     plt.xlabel("Zernike N",fontsize=25)
        #     plt.ylabel("value of Zernike coefficient",fontsize=25)
        #     plt.xticks([3,10,15,20,25,30,35,40,45,50,55,60,65],size = 20)
        #     plt.yticks(size=20)
        #     plt.rcParams.update({'font.size':25})
        #     plt.legend()
        #     plt.grid(ls='-.',linewidth=0.4)
        #     plt.savefig((dir+"real_predict.png"))
        #     plt.close()



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
        # print("x4",x4)
        # print([i for i in range (0,2000,200)])
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
        
        plt.figure(1, dpi=400)
        plt.contourf(real_out_all[ii-1,framnum,:,:],levels=[i*interval for i in range(int(np.max(real_out_all[ii-1,framnum,:,:])/interval + 2*interval))], cmap=plt.get_cmap('jet'))
        plt.colorbar()
        plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/2) for i in range(-2,3)],size=10)
        plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/2) for i in range(-2,3)],size=10)
        plt.xlabel("x (m)",fontsize=15)
        plt.ylabel("y (m)",fontsize=15)
        plt.savefig((dir+str(framnum)+"_outIntensity_real.png"),bbox_inches='tight')
        plt.close()

        plt.figure(1, dpi=400)
        plt.contourf(diff_out[ii-1,framnum,:,:],levels=[i*interval for i in range(int(np.max(diff_out[ii-1,framnum,:,:])/interval + 2*interval))], cmap=plt.get_cmap('jet'))
        plt.colorbar()
        plt.xticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/2) for i in range(-2,3)],size=10)
        plt.yticks([i*n_grid/4 for i in range(5)], ["%.4f" %(i*aaz/2) for i in range(-2,3)],size=10)
        plt.xlabel("x (m)",fontsize=15)
        plt.ylabel("y (m)",fontsize=15)
        plt.savefig((dir+str(framnum)+"outIntensity_diff.png"),bbox_inches='tight')
        plt.close()


    ######################### SR ######################################33333
    # if option_SR == 1:
    ideal = 1187.739988
    max_real = 0
    max_diff = 0
    for i in range(nsnapshots):
        max_diff = np.max(diff_out[ii-1,i,:,:])+max_diff
        max_real = np.max(real_out_all[ii-1,i,:,:])+max_real

    print("SR_diff = ",max_diff/nsnapshots/ideal)
    print("SR_real = ",max_real/nsnapshots/ideal)
    with open("/home/xianyuer/data/35_rms4_aaz36_200000/defocus/predict_b/SR_b.txt", encoding="utf-8",mode="a") as file:  
        file.write(str(max_diff/nsnapshots/ideal))
        file.write(" ")
        file.write(str(max_real/nsnapshots/ideal))
        file.write("\n")
###################################p########################
'''
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
value = np.zeros((nsnapshots,length,length))
r = np.zeros(nsnapshots)
p = np.zeros(nsnapshots)

real_0 = real_out_all
value_0 = np.zeros((nsnapshots,length,length))
r_0 = np.zeros(nsnapshots)
p_0 = np.zeros(nsnapshots)
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

    value_0[i,:,:] = real_0[i,:length, ]/np.max(real_0[i,:length, ])
    mass_center_0 = get_mass_center(value_0[i,:,:])
    # print("mass_center",mass_center)
    energy_profile_0 = get_energy_r(value_0[i,:,:], mass_center_0)
    # print(get_rcut(energy_profile, 0.84))
    rcut_0, intens_cut_0, energy_int_0 = get_rcut(energy_profile_0, 0.84)
    airy = 0.0061
    r_0[i] = rcut_0*airy*aaz0/256
    p_0[i] = r_0[i]/airy
np.savetxt(dir+"r_real.txt",p)

print("p_diff",np.mean(p))
print("p_real",np.mean(p_0))
with open("/home/xianyuer/data/35_rms4_aaz36_200000/rms4_aaz/p.txt", encoding="utf-8",mode="a") as file:  
    file.write(str(np.mean(p)))
    file.write(" ")
    file.write(str(np.mean(p_0)))
    file.write("\n")
# # plt.figure(1,dpi=300)
# # plt.plot(np.array(range(1,11)),r)
# # plt.xlabel("Frame M")
# # plt.ylabel("Radius(m)")
# # plt.savefig(dir+"radius.png",bbox_inches='tight')
'''