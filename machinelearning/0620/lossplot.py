import json
import matplotlib.pyplot as plt
import numpy as np
import math
with open("/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/10Zernike_test/0613_44_10000_64_0613_44_10000c_64_10000_100_history.json", "r") as json_file:
    json_dict44 = json.load(json_file)
    #print(json_dict)
    #print("type(json_dict) = >", type(json_dict))
    #print(json.dumps(json_dict, indent=4))

with open("/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/10Zernike_test/0613_104_10000_64_0613_104_10000c_64_10000_100_history.json", "r") as json_file:
    json_dict104 = json.load(json_file)
with open("/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/10Zernike_test/35_gauss_10000_64_0619_200-300epoch_10000_100_history.json", "r") as json_file:
    json_dict35 = json.load(json_file)
    #print(json_dict)
    #print("type(json_dict) = >", type(json_dict))
    #print(json.dumps(json_dict, indent=4))
with open("/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/10Zernike_test/0613_27_10000_64_0613_27_10000c_64_10000_100_history.json", "r") as json_file:
    json_dict27 = json.load(json_file)
with open("/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/10Zernike_test/0613_20_10000_64_0613_20_10000c_64_10000_100_history.json", "r") as json_file:
    json_dict20 = json.load(json_file)
with open("/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/10Zernike_test/0613_9_10000_64_0613_9_10000c_64_10000_100_history.json", "r") as json_file:
    json_dict9 = json.load(json_file)
with open("/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/10Zernike_test/0618_9_0.1_10000_64_0618_9_0.1_10000c_64_10000_200_history.json", "r") as json_file:
    json_dict9_01 = json.load(json_file)

plt.figure(1, dpi = 400)
plt.xlabel("Number of epoch in train set")
plt.ylabel("loss")
plt.plot(json_dict104['loss'], label = "104")
plt.plot(json_dict44['loss'], label = "44")
plt.plot(json_dict35['loss'], label = "35")
plt.plot(json_dict27['loss'], label = "27")
plt.plot(json_dict20['loss'], label = "20")
plt.plot(json_dict9['loss'], label = "9")
plt.legend()
#plt.legend(["loss_44","loss_27","loss_20","loss_9"])
plt.savefig("loss_0612_10000_64.png")
plt.close()

plt.figure(1, dpi = 400)
plt.xlabel("Number of epoch in train set")
plt.ylabel("val_loss")
plt.plot(json_dict104['val_loss'], label = "val_loss_104")
plt.plot(json_dict44['val_loss'], label = "val_loss_44")
plt.plot(json_dict35['val_loss'], label = "val_loss_35")
plt.plot(json_dict27['val_loss'], label = "val_loss_27")
plt.plot(json_dict20['val_loss'], label = "val_loss_20")
plt.plot(json_dict9['val_loss'], label = "val_loss_9")
plt.legend()
plt.savefig("val_loss_0612_10000_64.png")
plt.close()

loss_gauss = np.array(json_dict9['loss'])
loss_11 = np.array(json_dict9_01['loss'])
loss_11 = loss_11[100:200]

plt.figure(1, dpi = 400)
plt.plot(np.log10(loss_gauss), label = "gauss")
plt.plot(np.log10(loss_11), label = "[-0.1,0.1]")
#plt.legend()
plt.xlim(0,100)
plt.xlabel("Number of epoch in train set")
plt.ylabel("log10(loss)")
plt.legend()
plt.savefig("loss_distruntion.png")
plt.close()


