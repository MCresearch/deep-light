# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inPhase = np.loadtxt("/home/xianyuer/yuer/num/tests/inPhase.dat")

plt.figure(1, dpi = 300)
plt.contourf(inPhase)
plt.savefig("inPhase.png")
plt.close()