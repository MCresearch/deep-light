# import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from Zernike import *
from fun import *
import time
import json
import numba
from numba import jit
import torch

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    far_field_intens_orig = torch.zeros((nsnapshot,ngrid, ngrid)).to(device)
    down_intens = torch.zeros((nsnapshot,ngrid2, ngrid2)).to(device)
    gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))
    for iss in range(nsnapshot):
        phi0 = torch.zeros((ngrid, ngrid),dtype=torch.float).to(device)
        for i in range(ngrid):
            for j in range(ngrid):
                r2 = gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j]
                if r2/a02 <= 1:
                    phi0[i,j] = phi0[i,j] + torch.sum(Zer[i,j,3:]*cz[iss,:])
        # obj0_ = torch.zeros(ngrid,ngrid) + 1j*torch.zeros(ngrid,ngrid)   
              
        # propagation calculation
        obj0_ = init_intens*torch.exp(1j*phi0) # initial field

        dlta = (1-aaz/aa0)/zfh
        ddxz = 1-dlta*zfh
        dk0 = 1/aa0
        zzzz = zfh/(1-dlta*zfh)
        wave_number = 2*torch.pi/plm
        ################## focusing ###################
        ei = -wave_number*(gx*gx+gy*gy)/2*(1/zfh)
        ei = torch.tensor(ei).to(device)
        img0_ = obj0_*torch.exp(1j*ei) #focusing
   
        ############## mdfph #################
        ec = wave_number*gx*gx*dlta/2 + wave_number*gy*gy*dlta/2
        ec = torch.tensor(ec).to(device)
        img0_ = img0_*torch.exp(1j*ec) #mdfph

        ############# fft #################
        img0_ = torch.fft.fft2(img0_)
        img0_= torch.concat([\
        torch.concat([img0_[ngrid2:ngrid, ngrid2:ngrid], img0_[0:ngrid2, ngrid2:ngrid]], axis=0),\
        torch.concat([img0_[ngrid2:ngrid, 0:ngrid2], img0_[0:ngrid2, 0:ngrid2]], axis=0),\
                        ], axis=1) # far field
        ######## far field transmission ########
        h = torch.zeros(ngrid)
        prop1(ngrid,n1,zzzz,wave_number,aa0,h)
        h = torch.tensor(h).to(device)
        evol1(ngrid,h,img0_)
     
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
 
        # img0_.real[0,0]=-0.000105
        # img0_.real[0,1]=0.011373
        # img0_.real[1,0]=0.011373
        # img0_.real[1,1]=0.101484
        
        # img0_.imag[0,0]=0.000366
        # img0_.imag[0,1]=0.003268
        # img0_.imag[1,0]=0.003268
        # img0_.imag[1,1]=-0.353208	
        # print(img0_)
        ######## mdfph ############
        gy2,gx2 = np.meshgrid(dxyz*np.linspace(1-n1,ngrid-n1,ngrid),dxyz*np.linspace(1-n1,ngrid-n1,ngrid))
        ez = -1*wave_number*gx2*gx2*dlta/(2*ddxz) - wave_number*gy*gy*dlta/(2*ddxz)
        ez = torch.tensor(ez).to(device)
        img0_ = img0_*torch.exp(1j*ez) #mdfph
        img0_ = img0_/ddxz
        
        int_out = torch.abs(img0_)**2

        far_field_intens_orig[iss,:,:] = int_out
    return far_field_intens_orig

def nor_progagtion(nsnapshot,mm,a0,xx0,plm,zfh,xxz,init_intens,cz,Zer):
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    far_field_intens_orig = torch.zeros((nsnapshot,ngrid, ngrid)).to(device)
    down_intens = torch.zeros((nsnapshot,ngrid2, ngrid2)).to(device)
    gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))
    for iss in range(nsnapshot):
        phi0 = torch.zeros((ngrid, ngrid),dtype=torch.float).to(device)
        for i in range(ngrid):
            for j in range(ngrid):
                r2 = gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j]
                if r2/a02 <= 1:
                    phi0[i,j] = phi0[i,j] + torch.sum(Zer[i,j,3:]*cz[iss,:])
        # obj0_ = torch.zeros(ngrid,ngrid) + 1j*torch.zeros(ngrid,ngrid)   
              
        # propagation calculation
        obj0_ = init_intens*torch.exp(1j*phi0) # initial field

        dlta = (1-aaz/aa0)/zfh
        ddxz = 1-dlta*zfh
        dk0 = 1/aa0
        zzzz = zfh/(1-dlta*zfh)
        wave_number = 2*torch.pi/plm
        ################## focusing ###################
        ei = -wave_number*(gx*gx+gy*gy)/2*(1/zfh)
        ei = torch.tensor(ei).to(device)
        img0_ = obj0_*torch.exp(1j*ei) #focusing
   
        ############## mdfph #################
        ec = wave_number*gx*gx*dlta/2 + wave_number*gy*gy*dlta/2
        ec = torch.tensor(ec).to(device)
        img0_ = img0_*torch.exp(1j*ec) #mdfph

        ############# fft #################
        img0_ = torch.fft.fft2(img0_)
        img0_= torch.concat([\
        torch.concat([img0_[ngrid2:ngrid, ngrid2:ngrid], img0_[0:ngrid2, ngrid2:ngrid]], axis=0),\
        torch.concat([img0_[ngrid2:ngrid, 0:ngrid2], img0_[0:ngrid2, 0:ngrid2]], axis=0),\
                        ], axis=1) # far field
        ######## far field transmission ########
        h = torch.zeros(ngrid)
        prop1(ngrid,n1,zzzz,wave_number,aa0,h)
        evol1(ngrid,h,img0_)
     
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
 
        # img0_.real[0,0]=-0.000105
        # img0_.real[0,1]=0.011373
        # img0_.real[1,0]=0.011373
        # img0_.real[1,1]=0.101484
        
        # img0_.imag[0,0]=0.000366
        # img0_.imag[0,1]=0.003268
        # img0_.imag[1,0]=0.003268
        # img0_.imag[1,1]=-0.353208	
        # print(img0_)
        ######## mdfph ############
        gy2,gx2 = np.meshgrid(dxyz*np.linspace(1-n1,ngrid-n1,ngrid),dxyz*np.linspace(1-n1,ngrid-n1,ngrid))
        ez = -1*wave_number*gx2*gx2*dlta/(2*ddxz) - wave_number*gy*gy*dlta/(2*ddxz)
        ez = torch.tensor(ez).to(device)
        img0_ = img0_*torch.exp(1j*ez) #mdfph
        img0_ = img0_/ddxz
        
        int_out = torch.abs(img0_)**2
        far_field_intens_orig[iss,:,:] = int_out/torch.max(int_out)
    return far_field_intens_orig

def nor_down_progagtion(nsnapshot,mm,a0,xx0,plm,zfh,xxz,init_intens,cz,Zer):
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    far_field_intens_orig = torch.zeros((nsnapshot,ngrid, ngrid)).to(device)
    down_intens = torch.zeros((nsnapshot,ngrid2, ngrid2)).to(device)
    gy,gx = np.meshgrid(dxy0*np.linspace(1-n1,ngrid-n1,ngrid),dxy0*np.linspace(1-n1,ngrid-n1,ngrid))
    for iss in range(nsnapshot):
        phi0 = torch.zeros((ngrid, ngrid),dtype=torch.float).to(device)
        for i in range(ngrid):
            for j in range(ngrid):
                r2 = gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j]
                if r2/a02 <= 1:
                    phi0[i,j] = phi0[i,j] + torch.sum(Zer[i,j,3:]*cz[iss,:])
        # obj0_ = torch.zeros(ngrid,ngrid) + 1j*torch.zeros(ngrid,ngrid)   
              
        # propagation calculation
        obj0_ = init_intens*torch.exp(1j*phi0) # initial field

        dlta = (1-aaz/aa0)/zfh
        ddxz = 1-dlta*zfh
        dk0 = 1/aa0
        zzzz = zfh/(1-dlta*zfh)
        wave_number = 2*torch.pi/plm
        ################## focusing ###################
        ei = -wave_number*(gx*gx+gy*gy)/2*(1/zfh)
        ei = torch.tensor(ei).to(device)
        img0_ = obj0_*torch.exp(1j*ei) #focusing
   
        ############## mdfph #################
        ec = wave_number*gx*gx*dlta/2 + wave_number*gy*gy*dlta/2
        ec = torch.tensor(ec).to(device)
        img0_ = img0_*torch.exp(1j*ec) #mdfph

        ############# fft #################
        img0_ = torch.fft.fft2(img0_)
        img0_= torch.concat([\
        torch.concat([img0_[ngrid2:ngrid, ngrid2:ngrid], img0_[0:ngrid2, ngrid2:ngrid]], axis=0),\
        torch.concat([img0_[ngrid2:ngrid, 0:ngrid2], img0_[0:ngrid2, 0:ngrid2]], axis=0),\
                        ], axis=1) # far field
        ######## far field transmission ########
        h = torch.zeros(ngrid)
        prop1(ngrid,n1,zzzz,wave_number,aa0,h)
        h = torch.tensor(h).to(device)
        evol1(ngrid,h,img0_)
     
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
 
        # img0_.real[0,0]=-0.000105
        # img0_.real[0,1]=0.011373
        # img0_.real[1,0]=0.011373
        # img0_.real[1,1]=0.101484
        
        # img0_.imag[0,0]=0.000366
        # img0_.imag[0,1]=0.003268
        # img0_.imag[1,0]=0.003268
        # img0_.imag[1,1]=-0.353208	
        # print(img0_)
        ######## mdfph ############
        gy2,gx2 = np.meshgrid(dxyz*np.linspace(1-n1,ngrid-n1,ngrid),dxyz*np.linspace(1-n1,ngrid-n1,ngrid))
        ez = -1*wave_number*gx2*gx2*dlta/(2*ddxz) - wave_number*gy*gy*dlta/(2*ddxz)
        ez = torch.tensor(ez).to(device)
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