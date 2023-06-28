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


def parameter(mm,mgs,a0,xx0,plm,zfh,xxz,maxZnkOrder,minZnkDim,rms,eeznk,zernike_dir):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    # Transmission parameter calculation
    Zer,maxZnkDim = Zer1(maxZnkOrder,mm,a0,xx0)
    print(maxZnkDim)

    Zernike_alias_all = np.array([1] * 2 + [-1] * 3 + [1] * 4 + [-1] * 5 + [1] * 6 + [-1] * 7 + [1] * 8+ [-1] * 9+ [1] * 10 + [-1] * 11+ [1] * 12+ [-1] * 13 + [1] * 14, dtype=np.float32)
    Zernike_alias =  Zernike_alias_all[2:maxZnkDim]
    Zernike_alias  = torch.tensor(Zernike_alias).to(device)
    Zer,maxZnkDim = Zer1(maxZnkOrder,mm,a0,xx0)
    print("shape zer",np.shape(Zer))
    Zer = torch.tensor(Zer).to(device)

    init_intens = init_intensity(mm,a0,xx0,mgs)
    init_intens = torch.tensor(init_intens).to(device)

    nxzz = a0*xx0
    ngrid = pow(2,mm)
    n1 = ngrid/2 + 1
    aa0 = xx0*a0
    dxy0 = aa0/ngrid
    airy = 1.22*plm*zfh/(2*a0)
    aaz = airy*xxz
    dxyz = aaz/ngrid
    ngrid2 = ngrid//2
    a02 = a0*a0
    gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))
    gy2,gx2 = np.meshgrid(dxyz*np.linspace(1-n1,ngrid-n1,ngrid),dxyz*np.linspace(1-n1,ngrid-n1,ngrid))

    dlta = (1-aaz/aa0)/zfh
    ddxz = 1-dlta*zfh
    dk0 = 1/aa0
    zzzz = zfh/(1-dlta*zfh)
    wave_number = 2*torch.pi/plm
    ei = -wave_number*(gx*gx+gy*gy)/2*(1/zfh)
    ei = torch.tensor(ei).to(device)
    ec = wave_number*gx*gx*dlta/2 + wave_number*gy*gy*dlta/2
    ec = torch.tensor(ec).to(device)
    h = torch.zeros(ngrid).to(device)
    h_sum = torch.zeros((ngrid,ngrid)).to(device)
    prop1(ngrid,n1,zzzz,wave_number,aa0,h)
    for i in range(ngrid):
        for j in range(ngrid):
            h_sum[i,j] = h[i]+h[j]
            
    ez = -1*wave_number*gx2*gx2*dlta/(2*ddxz) - wave_number*gy2*gy2*dlta/(2*ddxz)
    ez = torch.tensor(ez).to(device)
    mask0 = ((gx**2 + gy**2)/a02 <= 1)
    mask0 = torch.tensor(mask0).to(device)
    f_m = torch.exp(1j*ei)*torch.exp(1j*ec)

    # gauss
    r02 = a0*a0*xx0*xx0/36
    gauss = np.ones((ngrid,ngrid))
    mask00 = ((gx*gx + gy*gy)<=r02)
    gauss  = gauss * mask00
    gauss  = torch.tensor(gauss).to(device)
    gauss = torch.reshape(gauss, [1, ngrid, ngrid]).to(device)  
    
    return Zernike_alias,maxZnkOrder,eeznk,rms,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss


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

# @jit(nopython=True)
def Zer(nsnapshot,maxZnkOrder,mm,a0,xx0,Phase_option,eeznk,rms,zernike_dir):
    
    ngrid = pow(2,mm)
    n1 = ngrid/2 + 1
    aa0 = xx0*a0
    dxy0 = aa0/ngrid
    a02 = a0*a0
    
    maxZnkDim = maxZernike(maxZnkOrder)
    print("maxZnkDim=",maxZnkDim)
    Zernike_order = []
    Zer = np.zeros((ngrid,ngrid,maxZnkDim+1))
    cz_ = np.zeros((nsnapshot,maxZnkDim))
    for iorder in range(0, maxZnkOrder+1):
        Zernike_order += [iorder for ii in range(iorder+1)]
    Zernike_order = np.array(Zernike_order)

    # generate random phase and its corresponding far field intensity
    gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))  
    for i in range(ngrid):
        for j in range(ngrid):
            r2 = gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j]
            if r2/a02 <= 1:
                Zer[i,j,:] = Zernike(maxZnkDim,gx[i,j]/a0,gy[i,j]/a0) # Zernike mode
    for iss in range(nsnapshot):
        if Phase_option == "random":   
            cz_[iss,2:] = np.random.normal(np.zeros(maxZnkDim-2), np.exp(-eeznk*(Zernike_order[3:]-1)))
            ss = np.sum(pow(cz_[iss,2:],2))
            cz_[iss,2:] *= np.sqrt(rms/ss) # normalization factor
    
    if Phase_option == "confirm":
        cz_[:,2:] = np.loadtxt(zernike_dir)
        print(cz_)
        
    return Zer,cz_

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
    return Zer,maxZnkDim

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


def progagtion(nsnapshot,ngrid,ngrid2,init_intens,cz,Zer,maxZnkDim ,mask0,f_m,h_sum,ez,ddxz):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    far_field_intens_orig = torch.zeros((nsnapshot,ngrid, ngrid)).to(device)
    down_intens = torch.zeros((nsnapshot,ngrid2, ngrid2)).to(device)
    for iss in range(nsnapshot):
        phi0 = torch.zeros((ngrid, ngrid),dtype=torch.float).to(device)
        phi0 = torch.sum(Zer[:,:,3:]*torch.reshape(cz[iss,:],[1,1,maxZnkDim-2]),2)
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

def sumnor_progagtion(nsnapshot,ngrid,ngrid2,init_intens,cz,Zer,maxZnkDim ,mask0,f_m,h_sum,ez,ddxz):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    far_field_intens_orig = torch.zeros((nsnapshot,ngrid, ngrid)).to(device)
    down_intens = torch.zeros((nsnapshot,ngrid2, ngrid2)).to(device)
    for iss in range(nsnapshot):
        phi0 = torch.zeros((ngrid, ngrid),dtype=torch.float).to(device)
        phi0 = torch.sum(Zer[:,:,3:]*torch.reshape(cz[iss,:],[1,1,maxZnkDim-2]),2)
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
        sum_1 = torch.sum(int_out)
        far_field_intens_orig[iss,:,:] = int_out/sum_1
    return far_field_intens_orig

def nor_down_progagtion(nsnapshot,ngrid,ngrid2,init_intens,cz,Zer,maxZnkDim ,mask0,f_m,h_sum,ez,ddxz):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    far_field_intens_orig = torch.zeros((nsnapshot,ngrid, ngrid)).to(device)
    down_intens = torch.zeros((nsnapshot,ngrid2, ngrid2)).to(device)
    for iss in range(nsnapshot):
        phi0 = torch.zeros((ngrid, ngrid),dtype=torch.float).to(device)
        phi0 = torch.sum(Zer[:,:,3:]*torch.reshape(cz[iss,:],[1,1,maxZnkDim-2]),2)
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
        for i in range(0,ngrid,2):
            for j in range(0,ngrid,2):
                max = int_out[i,j]
                if max < int_out[i,j+1]:
                    max = int_out[i,j + 1]
                if max < int_out[i+1,j]:
                    max = int_out[i+1,j]
                if max < int_out[i+1,j+1]:
                    max = int_out[i+1,j + 1]     
                down_intens[iss,i//2,j//2] = max
        down_intens[iss,:,:] = down_intens[iss,:,:]/torch.max(down_intens[iss,:,:])
    return down_intens