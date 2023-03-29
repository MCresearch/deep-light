import numpy as np
import matplotlib.pyplot as plt

data_1 = np.loadtxt("/home/xianyuer/yuer/testwej/deep-light/model_intensity_loss/repro/zer9_rms4/35_3456789_64_100_b1_gauss1loss_rms4_20000-40000.log")
data_2 = np.loadtxt("/home/xianyuer/yuer/testwej/deep-light/model_intensity_loss/repro/zer9_rms4/35_3456789_64_100_b1_gauss1loss_rms4_40000-60000.log")
data_3 = np.loadtxt("/home/xianyuer/yuer/testwej/deep-light/model_intensity_loss/repro/zer9_rms4/35_3456789_64_100_b1_gauss1loss_rms4_60000-100000.log")
data_4 = np.loadtxt("/home/xianyuer/yuer/testwej/deep-light/model_intensity_loss/repro/zer9_rms4/35_3456789_64_100_b1_gauss1loss_rms4_60000-100000_20000changeloss.log")
data_5 = np.loadtxt("/home/xianyuer/yuer/testwej/deep-light/model_intensity_loss/repro/zer9_rms4/35_3456789_64_100_b1_gauss1loss_rms4_60000-100000_40000changeloss.log")
data_6 = np.loadtxt("/home/xianyuer/yuer/testwej/deep-light/model_intensity_loss/repro/zer9_rms4/35_3456789_64_100_b1_gauss1loss_rms4_60000-100000_60000changeloss.log")
data_7 = np.loadtxt("/home/xianyuer/yuer/testwej/deep-light/model_intensity_loss/repro/zer9_rms4/35_3456789_64_100_b1_gauss1loss_rms4_60000-100000_80000changeloss.log")
data = np.concatenate((data_1,data_2))
data = np.concatenate((data,data_3))
data = np.concatenate((data,data_4))
data = np.concatenate((data,data_5))
data = np.concatenate((data,data_6))
data = np.concatenate((data,data_7))
print(np.shape(data))
#model_name="/home/xianyuer/yuer/testwej/deep-light/model_intensity_loss/repro/e1000_b8_testxception/"
model_name = "35_3456789_64_100_b1_gauss1loss_rms4_1-100000+80000changeloss"

x = np.zeros((1600))
for i in range(1600):
    x[i] = (i+1)*100+20000
plt.figure(1, dpi = 400)
plt.plot(x,data[:,0], color="red",alpha=1,label = "intensity loss")
# plt.xlabel("Zernike order",fontsize=15)
# plt.ylabel("Zernike coefficient values",fontsize=15)
# plt.xticks([i for i in range(1,51)])
# plt.yticks(size=10)
# # plt.title("Test set No.%d, model = 35_64_50_intloss" % 2,fontsize=15)
plt.legend()
plt.savefig(model_name+"loss_intloss.png",bbox_inches='tight') #,bbox_inches='tight'
plt.close()


plt.figure(1, dpi = 400)
plt.plot(x,data[:,1], color="blue",alpha=1,label = "Zernike loss(min)")
# plt.xlabel("Zernike order",fontsize=15)
# plt.ylabel("Zernike coefficient values",fontsize=15)
# plt.xticks([i for i in range(1,51)])
# plt.yticks(size=10)
# # plt.title("Test set No.%d, model = 35_64_50_intloss" % 2,fontsize=15)
plt.legend()
plt.savefig(model_name+"loss_zerloss.png",bbox_inches='tight') #,bbox_inches='tight'
plt.close()
