import json
import matplotlib.pyplot as plt
import numpy as np
import math
with open("/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/0620/xceptionfull_35_2_64_220620_10000_50_history.json", "r") as json_file:
    json_dict44 = json.load(json_file)

plt.figure(1, dpi = 400)
plt.xlabel("Number of epoch in train set")
plt.ylabel("loss")
plt.plot(json_dict44['loss'], label = "xceptionfull_[-2,2]_35_64")
plt.legend()
#plt.legend(["loss_44","loss_27","loss_20","loss_9"])
plt.savefig("loss_xceptionfull_2_64_35.png")
plt.close()

