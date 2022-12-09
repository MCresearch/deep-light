import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Zernike import *
from train_model import Net
import time


# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
train_step = "66e5"
model_path = "models/save_step%s.pt"%train_step
model = Net()

model.load_state_dict(torch.load(model_path))

# turn model and data on GPU device
model = model.to(device)
#cz = torch.tensor(cz).to(device)
far_field_intens_torch = torch.tensor(far_field_intens).to(device)

# model inference
cz_pred = np.zeros((nsnapshot, 33))