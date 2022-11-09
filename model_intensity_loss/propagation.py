# import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Zernike import *
from fun import *
import time
import json

def init_intensity(mm,a0,xx0,mgs):

    ngrid = pow(2,mm)
    n1 = ngrid/2 + 1
    aa0 = xx0*a0
    dxy0 = aa0/ngrid
    a02 = a0*a0
    
    init_intens = np.zeros((ngrid,ngrid))+1j*np.zeros((ngrid,ngrid))
    
    gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))
    mask0 = ((gx**2 + gy**2)<=a02)
    init_intens.real = np.exp(-1*pow(((gx*gx+gy*gy)/a02),mgs)) 
    init_intens = init_intens*mask0

    return init_intens

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
            cz_[iss,3:] = np.random.normal(np.zeros(maxZnkDim-2), np.exp(-eeznk*(Zernike_order[3:]-1)))
            ss = np.sum(pow(cz_[iss,3,:],2))
            cz_[iss,3:] *= np.sqrt(rms/ss) # normalization factor
    
    if Phase_option == "confirm":
        cz_[:,2:] = np.loadtxt(zernike_dir)
        print(cz_)
        
    return Zer,cz_

def progagtion(nsnapshot,mm,a0,xx0,plm,zfh,xxz,init_intens,cz,Zer):

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

    far_field_intens_orig = np.zeros((nsnapshot,ngrid, ngrid))
    far_field_intens = np.zeros((nsnapshot,ngrid2, ngrid2))
    down_intens = np.zeros((nsnapshot,ngrid2, ngrid2))
    cz_ = cz
    gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))
    for iss in range(nsnapshot):
        phi0 = np.zeros((ngrid, ngrid))
        for i in range(ngrid):
            for j in range(ngrid):
                r2 = gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j]
                if r2/a02 <= 1:
                    phi0[i,j] = phi0[i,j] + np.sum(Zer[i,j,3:]*cz_[iss,2:])
        # print("phi0",phi0) 
        # print("1111")         
        # propagation calculation
        obj0_ = init_intens*np.exp(1j*phi0) # initial field
        dlta = (1-aaz/aa0)/zfh
        ddxz = 1-dlta*zfh
        dk0 = 1/aa0
        zzzz = zfh/(1-dlta*zfh)
        wave_number = 2*np.pi/plm
        
        ################## focusing ###################
        ei = -wave_number*(gx*gx+gy*gy)/2*(1/zfh)
        img0_ = obj0_*np.exp(1j*ei) #focusing
        int_focusing = pow(img0_.imag,2)+pow(img0_.real,2)
        
        ############## mdfph #################
        ec = wave_number*gx*gx*dlta/2 + wave_number*gy*gy*dlta/2
        img0_ = img0_*np.exp(1j*ec) #mdfph
        int_mdfph = pow(img0_.imag,2)+pow(img0_.real,2)
        
        ############## fft #################
        img0_ = np.fft.fft2(img0_)
        
        img0_= np.concatenate([\
                np.concatenate([img0_[ngrid2:ngrid, ngrid2:ngrid], img0_[0:ngrid2, ngrid2:ngrid]], axis=0),\
                np.concatenate([img0_[ngrid2:ngrid, 0:ngrid2], img0_[0:ngrid2, 0:ngrid2]], axis=0),\
                                ], axis=1) # far field
    
        ######## far field transmission ########
        h = np.zeros(ngrid)
        prop1(ngrid,n1,zzzz,wave_number,aa0,h)
        evol1(ngrid,h,img0_)
        
        ######## fft ############
        img0_ = np.fft.ifft2(img0_)
        
        AA = np.ones((ngrid, ngrid))
        for i in range(ngrid):
            for j in range(ngrid):
                if(i%2==0):
                    if(j%2!=0):
                        AA[i,j] = -1.0    
                else:
                    if(j%2==0):
                        AA[i,j] = -1.0 
        img0_ = img0_*AA
        
        ######## mdfph ############
        gy2,gx2 = np.meshgrid(dxyz*np.linspace(1-n1,ngrid-n1,ngrid),dxyz*np.linspace(1-n1,ngrid-n1,ngrid))
        ec = -1*wave_number*gx2*gx2*dlta/(2*ddxz) - wave_number*gy*gy*dlta/(2*ddxz)
        img0_ = img0_*np.exp(1j*ec) #mdfph
        img0_ = img0_/ddxz
        
        int_out = np.abs(img0_)**2
        far_field_intens_orig[iss,:,:] = int_out
        ###### down sample ################## 
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
    
        far_field_intens = down_intens
            
        cz = np.float32(cz)
        far_field_intens = np.float32(far_field_intens)
    return far_field_intens
        # np.save("./data/zernike.npy",cz)
        # np.save("./data/far_field_intens.npy",far_field_intens)
        # np.save("./data/far_field_intens_orig.npy",far_field_intens_orig)
    

