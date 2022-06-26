import numpy as np
import matplotlib.pyplot as plt

def plt_intensity2(intensity, file_name, size=None, log=False):
    plt.figure(1, dpi = 400)
    #cmap = plt.colormaps("plasma")
    if size is not None:
        fig_size = intensity.shape[0]
        assert(size < fig_size)
        intensity = intensity[int((fig_size-size)/2):fig_size-int((fig_size-size)/2), int((fig_size-size)/2):fig_size-int((fig_size-size)/2)]
    if log:
        plt.contourf(np.log10(intensity), levels=np.arange(-6, 6, 1.5) ,cmap=plt.get_cmap('Spectral'))
        plt.colorbar()
    else:
        plt.contourf(intensity, cmap=plt.get_cmap('Spectral'))
        plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(file_name)
    plt.close()
    return


plt_intensity2(intensity=np.genfromtxt("fitted_intensity.txt"), file_name="test.png", size=512)