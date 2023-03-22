# !/usr/bin/env python
# -*- encoding: utf-8 -*- #
 
# ----------------------------------------------------------------------------------------------------------
# File Name:        Zernike.py
# Author:           Xianyuer
# Version:          1.0
# Created:          2023/03/22 19:26:36
# Description:      Main Function: Calculate Zernike functions
#                   Cross Reference: NONE
# Function List:    maxZernike(nk) -- Compute the number of categories of a given Zernike order
#                   Zernike(N,x,y) -- Calculate Zernike function Z_n^l(x,y)
# Input List:
#         <name>       <type>        <description>
#         __NONE__     ---           ---
# Output List:
#         <name>       <type>        <description>
#         __NONE__     ---           ---
# History: 
#         <author>     <version>     <time>			<description>
#         Xianyuer     1.0           2023/03/22		XXX
# ----------------------------------------------------------------------------------------------------------

#------------ Load Package ------------#

import numpy as np
import numba
from numba import jit


#------------ Define Functions ------------#

# Compute the number of categories of a given Zernike order
@jit(nopython=True)
def maxZernike(nk):
    return nk*(nk+3)//2

# Calculate Zernike function Z_n^l(x,y)
@jit(nopython=True)
def Zernike(N,x,y):

    norm_factor = np.sqrt(np.array([1, 4, 4] +
     [3, 6, 6] + [8 for i in range(4)] + [5, 10, 10, 10, 10] + [12 for i in range(6)] + [7] + [14 for i in range(6)] + [16 for i in range(8)] + 
     [9] + [18 for i in range(8)] + [20 for i in range(10)] + [11] + [22 for i in range(10)] + [24 for i in range(12)] + [13] + [26 for i in range(12)] + 
     [28 for i in range(14)]))
    Zer = np.zeros(N+1)
    for i in range(1,N+1):
        
        # 1
        if i == 1:
            Zer[1] = x # -1
        if i == 2:
            Zer[2] = y
        
        x2 = x**2
        y2 = y**2
        r = x2 + y2
        cos2 = x2 - y2
        sin2 = 2*x*y
        # 2
        if i == 3:
            Zer[3] = -1 + 2*r # 0
        if i == 4:
            Zer[4] = sin2 # 2
        if i == 5:
            Zer[5] = cos2 # -2
        
        cos3 = x2*x - 3*x*y2
        sin3 = -y2*y + 3*y*x2
        # 3
        if i == 6:
            Zer[6] = -2*y + 3*y*r # 1
        if i == 7:
            Zer[7] = -2*x + 3*x*r # -1
        if i == 8:
            Zer[8] = sin3 # 3
        if i == 9:
            Zer[9]= cos3 # -3
        
        # 4
        r2 = r**2
        cos4 = cos2**2 - sin2**2
        sin4 = 2*cos2*sin2
        if i == 10:
            Zer[10] = 1 + 6*r2 - 6*r
        if i == 11:
            Zer[11]= cos2*(4*r-3)
        if i == 12:
            Zer[12]= sin2*(4*r-3)
        if i == 13:
            Zer[13] = cos4
        if i == 14:
            Zer[14] = sin4
    
        # 5
        cos5 = cos2*cos3 - sin2*sin3
        sin5 = sin2*cos3 + sin3*cos2
        
        if i == 15:
            Zer[15] = (10*r2 - 12*r + 3)*x
        if i == 16:
            Zer[16] = (10*r2 - 12*r + 3)*y
        if i == 17:
            Zer[17] = (5*r-4)*cos3
        if i == 18:
            Zer[18] = (5*r-4)*sin3
        if i == 19:
            Zer[19] = cos5
        if i == 20:
            Zer[20] = sin5
    
        # 6
        r3 = r2*r
        cos6 = cos3**2 - sin3**2
        sin6 = 2*cos3*sin3
        if i == 21:
            Zer[21] = 20*r3 - 30*r2 + 12*r -1
        if i == 22:
            Zer[22] = (15*r2 - 20*r + 6)*sin2
        if i == 23:
            Zer[23] = (15*r2 - 20*r + 6)*cos2
        if i == 24:
            Zer[24] = (6*r - 5)*sin4
        if i == 25:
            Zer[25] = (6*r - 5)*cos4
        if i == 26:
            Zer[26] = sin6
        if i == 27:
            Zer[27] = cos6
    
        # 7
        cos7 = cos3*cos4 - sin3*sin4
        sin7 = sin3*cos4 + sin4*cos3
        if i == 28:
            Zer[28] = (35*r3 - 60*r2 + 30*r - 4)*y
        if i == 29:
            Zer[29] = (35*r3 - 60*r2 + 30*r - 4)*x
        if i == 30:
            Zer[30] = (21*r2 - 30*r + 10)*sin3
        if i == 31:
            Zer[31] = (21*r2 - 30*r + 10)*cos3
        if i == 32:
            Zer[32] = (7*r - 6)*sin5
        if i == 33:
            Zer[33] = (7*r - 6)*cos5
        if i == 34:
            Zer[34] = sin7
        if i == 35:
            Zer[35] = cos7

        # 8

        cos8 = cos4*cos4-sin4*sin4
        sin8 = 2*sin4*cos4

        if i == 36:
            Zer[36] = 70 * r * r * r * r - 140 * r * r * r + 90 * r * r - 20 * r + 1
        if i == 37:
            Zer[37] = (56 * r * r * r - 105 * r * r + 60 * r - 10) * cos2
        if i == 38:
            Zer[38] = (56 * r * r * r - 105 * r * r + 60 * r - 10) * sin2
        if i == 39:
            Zer[39] = (28 * r * r - 42 * r + 15) * cos4
        if i == 40:
            Zer[40] = (28 * r * r - 42 * r + 15) * sin4
        if i == 41:
            Zer[41] = (8 * r - 7) * cos6
        if i == 42:
            Zer[42] = (8 * r - 7) * sin6
        if i == 43:
            Zer[43] = cos8
        if i == 44:
            Zer[44] = sin8
        
        # 9
        cos9 = cos5*cos4 - sin5*sin4
        sin9 = sin5*cos4 + cos5*sin4
        if i == 45:
            Zer[45] = (126 * r * r * r * r - 280 * r * r * r + 210 * r * r - 60 * r + 5) * x
        if i == 46:
            Zer[46] = (126 * r * r * r * r - 280 * r * r * r + 210 * r * r - 60 * r + 5) * y
        if i == 47:
            Zer[47] = (84 * r * r * r - 168 * r * r + 105 * r - 20) * cos3
        if i == 48:
            Zer[48] = (84 * r * r * r - 168 * r * r + 105 * r - 20) * sin3
        if i == 49:
            Zer[49] = (36 * r * r - 56 * r + 21) * cos5
        if i == 50:
            Zer[50] = (36 * r * r - 56 * r + 21) * sin5
        if i == 51:
            Zer[51] = (9 * r - 8) * cos7
        if i == 52:
            Zer[52] = (9 * r - 8) * sin7
        if i == 53:
            Zer[53] = cos9
        if i == 54:
            Zer[54] = sin9
        # 10
        cos10 = cos5 * cos5 - sin5 * sin5
        sin10 = 2 * sin5 * cos5
        if i == 55:
            Zer[55] = (252 * pow(r, 5) - 630 * r * r * r * r + 560 * r * r * r -
                                    210 * r * r + 30 * r - 1)
        if i == 56:
            Zer[56] = (210 * r * r * r * r - 504 * r * r * r + 420 * r * r - 140 * r + 15) * sin2
        if i == 57:
            Zer[57] = (210 * r * r * r * r - 504 * r * r * r + 420 * r * r - 140 * r + 15) * cos2
        if i == 58:
            Zer[58] =  (120 * r * r * r - 252 * r * r + 168 * r - 35) * sin4
        if i == 59:
            Zer[59] = (120 * r * r * r - 252 * r * r + 168 * r - 35) * cos4
        if i == 60:
            Zer[60] = (45 * r * r - 72 * r + 28) * sin6
        if i == 61:
            Zer[61] = (45 * r * r - 72 * r + 28) * cos6
        if i == 62:
            Zer[62] = (10 * r - 9) * sin8
        if i == 63:
            Zer[63] = (10 * r - 9) * cos8
        if i == 64:
            Zer[64] = sin10
        if i == 65:
            Zer[65] = cos10

        # 11
        cos11 = cos6 * cos5 - sin6 * sin5
        sin11 = sin6 * cos5 + cos6 * sin5
        if i == 66:
            Zer[66] = (462 * pow(r, 5) - 1260 * r * r * r * r + 1260 * r * r * r - 560 * r * r +
                       105 * r - 6) * x
        if i == 67:
            Zer[67] = (462 * pow(r, 5) - 1260 * r * r * r * r + 1260 * r * r * r - 560 * r * r +
                       105 * r - 6) * y
        if i == 68:
            Zer[68] = (330 * r * r * r * r - 840 * r * r * r + 756 * r * r - 280 * r + 35) * sin3
        if i == 69:
            Zer[69] = (330 * r * r * r * r - 840 * r * r * r + 756 * r * r - 280 * r + 35) * cos3
        if i == 70:
            Zer[70] = (165 * r * r * r - 360 * r * r + 252 * r - 56) * sin5
        if i == 71:
            Zer[71] = (165 * r * r * r - 360 * r * r + 252 * r - 56) * cos5
        if i == 72:
            Zer[72] = (55 * r * r - 90 * r + 36) * sin7
        if i == 73:
            Zer[73] = (55 * r * r - 90 * r + 36) * cos7
        if i == 74:
            Zer[74] = (11 * r - 10) * sin9
        if i == 75:
            Zer[75] = (11 * r - 10) * cos9
        if i == 76:
            Zer[76] = sin11
        if i == 77:
            Zer[77] = cos11
        
        # 12
        cos12 = cos6 * cos6 - sin6 * sin6
        sin12 = 2 * sin6 * cos6
        if i == 78:
            Zer[78] = (924 * pow(r, 6) - 2772 * pow(r, 5) + 3150 * pow(r, 4) -
                                   1680 * pow(r, 3) + 420 * pow(r, 2) - 42 * r + 1)
        if i == 79:
            Zer[79] = (792 * pow(r, 5) - 2310 * pow(r, 4) + 2520 * pow(r, 3) - 1260 * pow(r, 2) +
                       280 * r - 21) * cos2
        if i == 80:
            Zer[80] = (792 * pow(r, 5) - 2310 * pow(r, 4) + 2520 * pow(r, 3) - 1260 * pow(r, 2) +
                       280 * r - 21) * sin2
        if i == 81:
            Zer[81] = (495 * r * r * r * r - 1320 * r * r * r + 1260 * r * r - 504 * r + 70) * cos4
        if i == 82:
            Zer[82] = (495 * r * r * r * r - 1320 * r * r * r + 1260 * r * r - 504 * r + 70) * sin4
        if i == 83:
            Zer[83] = (220 * r * r * r - 495 * r * r + 360 * r - 84) * cos6
        if i == 84:
            Zer[84] = (220 * r * r * r - 495 * r * r + 360 * r - 84) * sin6
        if i == 85:
            Zer[85] = (66 * r * r - 110 * r + 45) * cos8
        if i == 86:
            Zer[86] = (66 * r * r - 110 * r + 45) * sin8
        if i == 87:
            Zer[87] = (12 * r - 11) * cos10
        if i == 88:
            Zer[88] = (12 * r - 11) * sin10
        if i == 89:
            Zer[89] = cos12
        if i == 90:
            Zer[90] = sin12
            
        # 13
        cos13 = cos7 * cos6 - sin7 * sin6
        sin13 = sin7 * cos6 + cos7 * sin6
        if i == 91:
            Zer[91] = (1716 * pow(r, 6) - 5544 * pow(r, 5) + 6930 * r * r * r * r -
                       4200 * r * r * r + 1260 * r * r - 168 * r + 7) * x
        if i == 92:
            Zer[92] = (1716 * pow(r, 6) - 5544 * pow(r, 5) + 6930 * r * r * r * r -
                       4200 * r * r * r + 1260 * r * r - 168 * r + 7) * y
        if i == 93:
            Zer[93] = (1287 * pow(r, 5) - 3960 * r * r * r * r + 4620 * r * r * r - 2520 * r * r +
                       630 * r - 56) * cos3
        if i == 94:
            Zer[94] = (1287 * pow(r, 5) - 3960 * r * r * r * r + 4620 * r * r * r - 2520 * r * r +
                       630 * r - 56) * sin3
        if i == 95:
            Zer[95] = (715 * r * r * r * r - 1980 * r * r * r + 1980 * r * r - 840 * r + 126) * cos5
        if i == 96:
            Zer[96] =  (715 * r * r * r * r - 1980 * r * r * r + 1980 * r * r - 840 * r + 126) * sin5
        if i == 97:
            Zer[97] = (286 * r * r * r - 660 * r * r + 495 * r - 120) * cos7
        if i == 98:
            Zer[98] = (286 * r * r * r - 660 * r * r + 495 * r - 120) * sin7
        if i == 99:
            Zer[99] = (78 * r * r - 132 * r + 55) * cos9
        if i == 100:
            Zer[100] = (78 * r * r - 132 * r + 55) * sin9
        if i == 101:
            Zer[101] = (13 * r - 12) * cos11
        if i == 102:
            Zer[102] = (13 * r - 12) * sin11
        if i == 103:
            Zer[103] = cos12
        if i == 104:
            Zer[104] = sin13
       

    Zer = Zer * norm_factor[:N+1]
    return Zer



    
