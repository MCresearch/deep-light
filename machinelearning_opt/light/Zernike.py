import numpy as np
def maxZernike(nk):

    maxZernike = 0
    l = 0
    ni = 0
    m0 = 0
    m1 = 0
    m = 0
    for ni in range(1,nk+1):
        
        m0 = ni - 2 * (int)(ni / 2)
        m1 = m0 + 2 * (int)(ni / 2)
        for m in range(m0,m1+1,2):
            l = l + 1
            if m != 0:
                l = l+1
    return l


def Zernike(N, x,y):

    norm_factor = np.sqrt(np.array([1, 4, 4] +
     [3, 6, 6] + [8 for i in range(4)] + [5, 10, 10, 10, 10] + [12 for i in range(6)] + [7] + [14 for i in range(6)] + [16 for i in range(8)]))
    Zer = np.zeros([N+1])
    
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
    
    Zer = Zer * norm_factor[:N+1]
    return Zer



    
