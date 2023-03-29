import json
import matplotlib.pyplot as plt
import numpy as np
import math
with open("../../data/65_256_1-10_midloop4_230222_2000_10_history.json", "r") as json_file:
    json_dict44 = json.load(json_file)

plt.figure(1, dpi = 400)
plt.xlabel("Number of epoch in train set")
plt.ylabel("loss")
plt.plot(json_dict44['loss'], label = "xceptionfull_[-2,2]_65_256")
plt.legend()
#plt.legend(["loss_44","loss_27","loss_20","loss_9"])
plt.savefig("../../data/result/loss_xceptionfull_2_256_65.png")
plt.close()

