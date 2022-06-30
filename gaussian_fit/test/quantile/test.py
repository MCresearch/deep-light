import numpy as np
intensity = np.genfromtxt("../gaussian/outIntensity.dat")
print(np.max(intensity))