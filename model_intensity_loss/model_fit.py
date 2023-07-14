#导入包
import numpy as np
import sys
# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset,DataLoader
from torchinfo import summary
sys.path.append("..")
import matplotlib.pyplot as plt
import sys
from Zernike import *
from fun import *
import time
import json
from propagation import *
import random
from Xception import *


def fit(model_name,net,loss_type,save,batch_size,epochs,learning_rate,print_step,save_step,zernike_dir,Zernike_alias,maxZnkOrder,eeznk,rms,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss_down):
    fid = open(model_name+'.log', 'w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate) 
    Zernike_alias_0 = torch.zeros((batch_size,maxZnkDim-2)).to(device)  
    gauss_0 = torch.zeros((batch_size,1,ngrid,ngrid)).to(device)  
    loss_z1 =  torch.zeros((batch_size)).to(device) 
    loss_z2 =  torch.zeros((batch_size)).to(device) 
    loss_zer =  torch.zeros((batch_size)).to(device)
    for i in range(batch_size):
        Zernike_alias_0[i,:] = Zernike_alias
        gauss_0[i,0,:,:] = gauss_down
    # Downsampling function
    # pool = torch.nn.MaxPool2d(
    #     2, 
    #     stride=None, 
    #     padding=0, 
    #     dilation=1, 
    #     return_indices=False, 
    #     ceil_mode=False)
    # scheduler = optim.lr_scheduler.LinearLR(opt,start_factor=0.00001, total_iters=epochs)
    #loss
    criterion = torch.nn.MSELoss(reduction='mean')  
    # Monitor progress
    samples = 0
    # Monitoring accuracy
    corrects = 0
    # Full data training several times
    for epoch in range(epochs):
        # data
        if zernike_dir =="random": 
            y = cc(batch_size,maxZnkOrder,"random",eeznk,rms,zernike_dir)
            y = y[:,2:]
        else:
            y = np.loadtxt(zernike_dir)
            
        y = torch.tensor(y).to(device)
        x = sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,y,Zer,maxZnkDim ,mask0,f_m,h_sum,ez,ddxz)
        x_c = x.clone()
        if save == "Ture":
            np.save("sumnor_outintensity.npy",x_c.detach().cpu().numpy())
        x = torch.reshape(x, [batch_size, 1, ngrid, ngrid]).to(device)  
        x = x * gauss_0
        
        # forward propagation
        cz_pred = net(x)

        #  loss
        if loss_type == "Zernikeloss":
            cz_pred = cz_pred.to(torch.float32)
            y = y.to(torch.float32)
            loss_zer = criterion(cz_pred,y)
            loss = loss_zer
        elif loss_type == "intensityloss":
            far_field_intens_pred =  sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,cz_pred,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz)
            far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, ngrid, ngrid]).to(device)  
            far_field_intens_pred = far_field_intens_pred * gauss_0
            loss_int = torch.mean(pow(far_field_intens_pred - x,2))
            loss = loss_int
        elif loss_type == "Zernike+intensityloss":  
            cz_pred = cz_pred.to(torch.float32)
            y = y.to(torch.float32)
            loss_zer = criterion(cz_pred,y)
            far_field_intens_pred =  sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,cz_pred,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz)
            far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, ngrid, ngrid]).to(device)  
            far_field_intens_pred = far_field_intens_pred * gauss_0
            loss_int = torch.mean(pow(far_field_intens_pred - x,2))
            loss = loss_int + loss_zer
            
        else:
            print("error loss_type", flush=True)
        # Gradient clearing
        opt.zero_grad()
        # backpropagation
        loss.backward()
        # Renewal gradient
        opt.step()
        
        # 监视进度：每训练一个batch，模型见过的数据就会增加x.shape[0]
        samples += x.shape[0]
        # print
        if (epoch + 1) % (print_step) == 0 or epoch  == (epochs - 1):
            loss_z1 = torch.mean(pow(cz_pred - y,2))
            loss_z2 = torch.mean(pow(cz_pred*Zernike_alias_0 - y,2))
            loss_z1_c = loss_z1.clone()
            loss_z2_c = loss_z2.clone()
            loss_zernike = np.minimum(loss_z1_c.detach().cpu().numpy(), loss_z2_c.detach().cpu().numpy()) 
            # loss_zernike = np.mean(pow(cz_pred.detach().cpu().numpy() - y.detach().cpu().numpy(),2))
            print("Epoch{}:[{}/{} {: .0f}%], Loss:{:.6f} ".format(
                epoch + 1
                , samples
                , epochs*batch_size
                , print_step*samples/(epochs*batch_size)
                , loss.data.item()),"Zer loss: ",loss_zernike, flush=True)
            fid.write(str(loss.cpu().item())+'\t'+str(loss_zernike))
            fid.write('\n')
        if (epoch + 1) % (save_step) == 0 or epoch  == (epochs - 1):
            torch.save(net.state_dict(), model_name+"_step_"+str(epoch + 1)+"_lr_"+str(learning_rate)+'.pt')


def fit2(batchdata,model_name,net,loss_type,save,batch_size,epochs,learning_rate,print_step,save_step,zernike_dir,Zernike_alias,maxZnkOrder,eeznk,rms,ngrid,ngrid2,init_intens,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz,gauss_down):
    fid = open(model_name+'.log', 'w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate) 
    Zernike_alias_0 = torch.zeros((batch_size,maxZnkDim-2)).to(device)  
    gauss_0 = torch.zeros((batch_size,1,ngrid,ngrid)).to(device)  
    loss_z1 =  torch.zeros((batch_size)).to(device) 
    loss_z2 =  torch.zeros((batch_size)).to(device) 
    loss_zer =  torch.zeros((batch_size)).to(device)
    for i in range(batch_size):
        Zernike_alias_0[i,:] = Zernike_alias
        gauss_0[i,0,:,:] = gauss_down
    #loss
    criterion = torch.nn.MSELoss(reduction='mean')  
    # Monitor progress
    samples = 0
    # Monitoring accuracy
    corrects = 0
    # Full data training several times
    for epoch in range(epochs):
        # 对每个batch进行训练
        for batch_idx, (x,y) in enumerate(batchdata):
            # forward propagation
            x = x.half().float()
            cz_pred = net(x)
            #  loss
            if loss_type == "Zernikeloss":
                cz_pred = cz_pred.to(torch.float32)
                y = y.to(torch.float32)
                loss_zer = criterion(cz_pred,y)
                loss = loss_zer
            elif loss_type == "intensityloss":
                far_field_intens_pred =  sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,cz_pred,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz)
                far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, ngrid, ngrid]).to(device)  
                far_field_intens_pred = far_field_intens_pred 
                loss_int = torch.mean(pow(far_field_intens_pred - x,2))
                loss = loss_int
            elif loss_type == "Zernike+intensityloss":  
                cz_pred = cz_pred.to(torch.float32)
                y = y.to(torch.float32)
                loss_zer = criterion(cz_pred,y)
                far_field_intens_pred =  sumnor_progagtion(batch_size,ngrid,ngrid2,init_intens,cz_pred,Zer,maxZnkDim,mask0,f_m,h_sum,ez,ddxz)
                far_field_intens_pred = torch.reshape(far_field_intens_pred, [batch_size, 1, ngrid, ngrid]).to(device)  
                far_field_intens_pred = far_field_intens_pred
                loss_int = torch.mean(pow(far_field_intens_pred - x,2))
                loss = loss_int + loss_zer 
            else:
                print("error loss_type", flush=True)
            # backpropagation
            loss.backward()
            # Renewal gradient
            opt.step()
            # Gradient clearing
            opt.zero_grad()
        
            # 监视进度：每训练一个batch，模型见过的数据就会增加x.shape[0]
            samples += x.shape[0]
            # print
            if (batch_idx + 1) % (print_step) == 0 or batch_idx == (len(batchdata) - 1):
                loss_z1 = torch.mean(pow(cz_pred - y,2))
                loss_z2 = torch.mean(pow(cz_pred*Zernike_alias_0 - y,2))
                loss_z1_c = loss_z1.clone()
                loss_z2_c = loss_z2.clone()
                loss_zernike = np.minimum(loss_z1_c.detach().cpu().numpy(), loss_z2_c.detach().cpu().numpy()) 
                # loss_zernike = np.mean(pow(cz_pred.detach().cpu().numpy() - y.detach().cpu().numpy(),2))
                print("Epoch{}:[{}/{} {: .0f}%], Loss:{:.3e} ".format(
                    epoch + 1
                    , samples
                    , epochs*len(batchdata.dataset)
                    , 100*samples/(epochs*len(batchdata.dataset))
                    , loss.data.item()),"Zer loss:{:.3e} ",loss_zernike, flush=True)
                fid.write(str(loss.cpu().item())+'\t'+str(loss_zernike))
                fid.write('\n')
        if (epoch + 1) % (save_step) == 0 or epoch  == (epochs - 1):
            torch.save(net.state_dict(), model_name+"_step_"+str(epoch + 1)+"_lr_"+str(learning_rate)+'.pt')