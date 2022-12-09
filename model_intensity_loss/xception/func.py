#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:53:26 2020

@author: LiuRenxi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def shuffle(a, b):
    a = list(a)
    b = list(b)
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    a = np.array(a)
    b = np.array(b)
    return a, b    

def down_sample(x, stride=2):
    # the function takes a three-dimension array (nsnapshots, nx, ny). It downsamples the array to a (nsnapshots, nx/stride, ny/stride) one by choosing the maximum grid in each stride*stride little square.
    (nsnapshots, nx, ny) = x.shape
    nx1 = int(nx/stride)
    ny1 = int(ny/stride)
    print(nx1, ny1)
    new_x = np.zeros([nsnapshots, nx1, ny1])
    for isnapshot in range(nsnapshots):
        for ix in range(nx1):
            for iy in range(ny1):
                #print(iy*stride)
                #print(iy*stride+stride)
                startx = ix*stride
                endx = ix*stride+stride
                starty = iy*stride
                endy = iy*stride+stride
                #print(startx, endx, starty, endy)
                #print(x[isnapshot, startx:endx, starty:endy])
                new_x[isnapshot][ix][iy] = np.max(x[isnapshot, startx:endx, starty:endy])
        print(isnapshot)
    return new_x
