import numpy as np
import matplotlib.pyplot as plt
step = 0.00001
dzernike_back00 = np.loadtxt("dzernike_back_00_0.txt")
dzernike = np.loadtxt("dzernike_00_0.txt")
sumnor_outintensity = np.load("sumnor_outintensity_00_0.npy")
dzernike_0 = dzernike[:,0]

N = len(dzernike_0)
grad = np.zeros_like(dzernike_0[0:999])
for i in range(N-2):
    grad[i] = (dzernike_0[i+2] - dzernike_0[i])/step/2
    
# d_dzernike_0 = np.diff(dzernike_0)/0.0001
x = np.zeros((1000))
for i in range(1000):
    x[i] = sumnor_outintensity[0,0,0]+i*step

plt.figure(1,dpi=600)
plt.plot(x[:999],dzernike_back00[1:],label="backward")
# plt.plot(x,dzernike_0,label="dzernike_0")
plt.plot(x[:999],grad,label="numerical differentiation")
plt.suptitle('Backpropagation verification(intensity00_zernike0)',fontsize=15)
plt.legend()
plt.savefig("Backpropagation verification_intensity00_zernike0.png")