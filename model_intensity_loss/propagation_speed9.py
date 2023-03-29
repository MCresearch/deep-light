# import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from Zernike import *
from fun import *
import time
import json
from numba import jit
import torch
import torch.nn as nn
def init_intensity(mm,a0,xx0,mgs):

    ngrid = pow(2,mm)
    n1 = ngrid/2 + 1
    aa0 = xx0*a0
    dxy0 = aa0/ngrid
    a02 = a0*a0
    
    init_intens = np.zeros((ngrid,ngrid))+1j*np.zeros((ngrid,ngrid))
    gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))
    mask0 = ((gx*gx + gy*gy)<=a02)
    init_intens.real = np.exp(-1*pow(((gx*gx+gy*gy)/a02),mgs)) 
    init_intens = init_intens*mask0

    return init_intens

# # @jit(nopython=True)
# def Zer(nsnapshot,maxZnkOrder,mm,a0,xx0,Phase_option,eeznk,rms,zernike_dir):
    
#     ngrid = pow(2,mm)
#     n1 = ngrid/2 + 1
#     aa0 = xx0*a0
#     dxy0 = aa0/ngrid
#     a02 = a0*a0
    
#     maxZnkDim = maxZernike(maxZnkOrder)
#     print("maxZnkDim=",maxZnkDim)
#     Zernike_order = []
#     Zer = np.zeros((ngrid,ngrid,maxZnkDim+1))
#     cz_ = np.zeros((nsnapshot,maxZnkDim))
#     for iorder in range(0, maxZnkOrder+1):
#         Zernike_order += [iorder for ii in range(iorder+1)]
#     Zernike_order = np.array(Zernike_order)

#     # generate random phase and its corresponding far field intensity
#     gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))  
#     for i in range(ngrid):
#         for j in range(ngrid):
#             r2 = gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j]
#             if r2/a02 <= 1:
#                 Zer[i,j,:] = Zernike(maxZnkDim,gx[i,j]/a0,gy[i,j]/a0) # Zernike mode
#     for iss in range(nsnapshot):
#         if Phase_option == "random":   
#             cz_[iss,2:] = np.random.normal(np.zeros(maxZnkDim-2), np.exp(-eeznk*(Zernike_order[3:]-1)))
#             ss = np.sum(pow(cz_[iss,2:],2))
#             cz_[iss,2:] *= np.sqrt(rms/ss) # normalization factor
    
#     if Phase_option == "confirm":
#         cz_[:,2:] = np.loadtxt(zernike_dir)
#         print(cz_)
        
#     return Zer,cz_

def Zer1(maxZnkOrder,mm,a0,xx0):
    
    ngrid = pow(2,mm)
    n1 = ngrid/2 + 1
    aa0 = xx0*a0
    dxy0 = aa0/ngrid
    a02 = a0*a0
    
    maxZnkDim = maxZernike(maxZnkOrder)
    print("maxZnkDim=",maxZnkDim)
    Zer = np.zeros((ngrid,ngrid,maxZnkDim+1))
    # generate random phase and its corresponding far field intensity
    gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))  
    for i in range(ngrid):
        for j in range(ngrid):
            r2 = gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j]
            if r2/a02 <= 1:
                Zer[i,j,:] = Zernike(maxZnkDim,gx[i,j]/a0,gy[i,j]/a0) # Zernike mode
    return Zer

def cc(nsnapshot,maxZnkOrder,Phase_option,eeznk,rms,zernike_dir):
    maxZnkDim = maxZernike(maxZnkOrder)
    Zernike_order = []
    cz_ = np.zeros((nsnapshot,maxZnkDim))
    for iorder in range(0, maxZnkOrder+1):
        Zernike_order += [iorder for ii in range(iorder+1)]
    Zernike_order = np.array(Zernike_order)

    for iss in range(nsnapshot):
        if Phase_option == "random":   
            cz_[iss,2:] = np.random.normal(np.zeros(maxZnkDim-2), np.exp(-eeznk*(Zernike_order[3:]-1)))
            ss = np.sum(pow(cz_[iss,2:],2))
            cz_[iss,2:] *= np.sqrt(rms/ss) # normalization factor
    
    if Phase_option == "confirm":
        cz_[:,2:] = np.loadtxt(zernike_dir)
        print(cz_)
        
    return cz_


def progagtion(nsnapshot,ngrid,ngrid2,init_intens,cz,Zer,mask0,f_m,h_sum,ez,ddxz):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    far_field_intens_orig = torch.zeros((nsnapshot,ngrid, ngrid)).to(device)
    down_intens = torch.zeros((nsnapshot,ngrid2, ngrid2)).to(device)
    for iss in range(nsnapshot):
        phi0 = torch.zeros((ngrid, ngrid),dtype=torch.float).to(device)
        phi0 = torch.sum(Zer[:,:,3:]*torch.reshape(cz[iss,:],[1,1,33]),2)
        obj0_ = mask0*init_intens*torch.exp(1j*phi0) # initial field
        ################## focusing ###################
        # img0_ = obj0_*torch.exp(1j*ei) #focusing
        # ############## mdfph #################
        # img0_ = img0_*torch.exp(1j*ec) #mdfph
        img0_ = obj0_*f_m
        ############# fft #################
        img0_ = torch.fft.fft2(img0_)
        img0_= torch.concat([\
        torch.concat([img0_[ngrid2:ngrid, ngrid2:ngrid], img0_[0:ngrid2, ngrid2:ngrid]], axis=0),\
        torch.concat([img0_[ngrid2:ngrid, 0:ngrid2], img0_[0:ngrid2, 0:ngrid2]], axis=0),\
                        ], axis=1) # far field
        ######## far field transmission ########
        # evol1(ngrid,h,img0_)
        img0_ = img0_*torch.exp(1j*h_sum)
        ####### fft ############
        img0_ = torch.fft.ifft2(img0_)
        AA = np.ones((ngrid, ngrid))
        for i in range(ngrid):
            for j in range(ngrid):
                if(i%2==0):
                    if(j%2!=0):
                        AA[i,j] = -1.0    
                else:
                    if(j%2==0):
                        AA[i,j] = -1.0 
        AA = torch.tensor(AA).to(device)
        img0_ = img0_*AA
        ######## mdfph ############
        img0_ = img0_*torch.exp(1j*ez) #mdfph
        img0_ = img0_/ddxz
        int_out = torch.abs(img0_)**2
        far_field_intens_orig[iss,:,:] = int_out
    return far_field_intens_orig

def nor_progagtion(nsnapshot,ngrid,ngrid2,init_intens,cz,Zer,mask0,f_m,h_sum,ez,ddxz):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    far_field_intens_orig = torch.zeros((nsnapshot,ngrid, ngrid)).to(device)
    down_intens = torch.zeros((nsnapshot,ngrid2, ngrid2)).to(device)
    for iss in range(nsnapshot):
        phi0 = torch.zeros((ngrid, ngrid),dtype=torch.float).to(device)
        phi0 = torch.sum(Zer[:,:,3:]*torch.reshape(cz[iss,:],[1,1,33]),2)
        obj0_ = mask0*init_intens*torch.exp(1j*phi0) # initial field
        ################## focusing ###################
        # img0_ = obj0_*torch.exp(1j*ei) #focusing
        # ############## mdfph #################
        # img0_ = img0_*torch.exp(1j*ec) #mdfph
        img0_ = obj0_*f_m
        ############# fft #################
        img0_ = torch.fft.fft2(img0_)
        img0_= torch.concat([\
        torch.concat([img0_[ngrid2:ngrid, ngrid2:ngrid], img0_[0:ngrid2, ngrid2:ngrid]], axis=0),\
        torch.concat([img0_[ngrid2:ngrid, 0:ngrid2], img0_[0:ngrid2, 0:ngrid2]], axis=0),\
                        ], axis=1) # far field
        ######## far field transmission ########
        # evol1(ngrid,h,img0_)
        img0_ = img0_*torch.exp(1j*h_sum)
        ####### fft ############
        img0_ = torch.fft.ifft2(img0_)
        AA = np.ones((ngrid, ngrid))
        for i in range(ngrid):
            for j in range(ngrid):
                if(i%2==0):
                    if(j%2!=0):
                        AA[i,j] = -1.0    
                else:
                    if(j%2==0):
                        AA[i,j] = -1.0 
        AA = torch.tensor(AA).to(device)
        img0_ = img0_*AA
        ######## mdfph ############
        img0_ = img0_*torch.exp(1j*ez) #mdfph
        img0_ = img0_/ddxz
        int_out = torch.abs(img0_)**2
        far_field_intens_orig[iss,:,:] = int_out/torch.max(int_out)
    return far_field_intens_orig

def nor_down_progagtion(nsnapshot,ngrid,ngrid2,init_intens,cz,Zer,mask0,f_m,h_sum,ez,ddxz):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    far_field_intens_orig = torch.zeros((nsnapshot,ngrid, ngrid)).to(device)
    down_intens = torch.zeros((nsnapshot,ngrid2, ngrid2)).to(device)
    for iss in range(nsnapshot):
        phi0 = torch.zeros((ngrid, ngrid),dtype=torch.float).to(device)
        phi0 = torch.sum(Zer[:,:,3:]*torch.reshape(cz[iss,:],[1,1,33]),2)
        obj0_ = mask0*init_intens*torch.exp(1j*phi0) # initial field
        ################## focusing ###################
        # img0_ = obj0_*torch.exp(1j*ei) #focusing
        # ############## mdfph #################
        # img0_ = img0_*torch.exp(1j*ec) #mdfph
        img0_ = obj0_*f_m
        ############# fft #################
        img0_ = torch.fft.fft2(img0_)
        img0_= torch.concat([\
        torch.concat([img0_[ngrid2:ngrid, ngrid2:ngrid], img0_[0:ngrid2, ngrid2:ngrid]], axis=0),\
        torch.concat([img0_[ngrid2:ngrid, 0:ngrid2], img0_[0:ngrid2, 0:ngrid2]], axis=0),\
                        ], axis=1) # far field
        ######## far field transmission ########
        # evol1(ngrid,h,img0_)
        img0_ = img0_*torch.exp(1j*h_sum)
        ####### fft ############
        img0_ = torch.fft.ifft2(img0_)
        AA = np.ones((ngrid, ngrid))
        for i in range(ngrid):
            for j in range(ngrid):
                if(i%2==0):
                    if(j%2!=0):
                        AA[i,j] = -1.0    
                else:
                    if(j%2==0):
                        AA[i,j] = -1.0 
        AA = torch.tensor(AA).to(device)
        img0_ = img0_*AA
        ######## mdfph ############
        img0_ = img0_*torch.exp(1j*ez) #mdfph
        img0_ = img0_/ddxz
        int_out = torch.abs(img0_)**2
        far_field_intens_orig[iss,:,:] = int_out
        max = 0
        # for i in range(0,ngrid,2):
        #     for j in range(0,ngrid,2):
        #         max = int_out[i,j]
        #         if max < int_out[i,j+1]:
        #             max = int_out[i,j + 1]
        #         if max < int_out[i+1,j]:
        #             max = int_out[i+1,j]
        #         if max < int_out[i+1,j+1]:
        #             max = int_out[i+1,j + 1]     
        #         down_intens[iss,i//2,j//2] = max
        down_intens[iss,:,:] = down_intens[iss,:,:]/torch.max(down_intens[iss,:,:])
    return down_intens