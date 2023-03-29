# !/usr/bin/env python
# -*- encoding: utf-8 -*- #
 
# ----------------------------------------------------------------------------------------------------------
# File Name:        TestPlot
# Author:           Erjie Wu
# Version:          1.0
# Created:          2023/03/27 10:30:54
# Description:      Main Function: Some plot function for model test
#                   Cross Reference: 
# Function List:    IntCom(real,predict,diff,name) -- Plot the comparison of intensity distribution
#                   ZerCom(dimension,real,predict) -- Plot the comparison of Zernike coefficient
# Input List:
#         <name>       <type>        <description>
#         __NONE__     ---           ---
# Output List:
#         <name>       <type>        <description>
#         __NONE__     ---           ---
# History: 
#         <author>     <version>     <time>			<description>
#         Erjie Wu     1.0           2023/03/27		Creat the file, add IntCom and ZerCom
# ----------------------------------------------------------------------------------------------------------

#------------ Load Package ------------#

import numpy as np
import matplotlib.pyplot as plt


#------------ Define Functions ------------#

# Plot the comparison of intensity distribution
def IntCom(real,predict,diff,name):
    # REQUIRE real,predict,diff are 2D data
    plt.figure(1, figsize=(16,4))
    plt.subplot(131)
    plt.contourf(real.detach().cpu().numpy(),levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)

    plt.subplot(132)
    plt.contourf(predict.detach().cpu().numpy(),levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)

    plt.subplot(133)
    plt.contourf(diff.detach().cpu().numpy(),levels=[0.01*i for i in range(102)], cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xlabel("x (m)",fontsize=15)
    plt.ylabel("y (m)",fontsize=15)

    plt.savefig(name,bbox_inches='tight')
    plt.close()


# Plot the comparison of Zernike coefficient
def ZerCom(dimension,real,predict):
    # REQUIRE real,predict are 1D data
    plt.bar(dimension,real.detach().cpu().numpy(), color="red",alpha=1,label = "Initial values")
    plt.bar(dimension,predict.detach().cpu().numpy(), color="blue",alpha=0.5,label = "Predict values")
    plt.xlabel("Zernike order",fontsize=15)
    plt.ylabel("Zernike coefficient values",fontsize=15)
    plt.xticks(size = 10)
    plt.yticks(size = 10)
    plt.legend()
