# 生成并存储zernike系数文件
import numpy as np
import os
nsnapshots = 10000 # 指定帧数
nbatch = 1 # 分成nbatch个dat文件生成
nzernike = 65 # zernike阶数
low = -0.5 # zernike分布下限
high = 0.5 # zernike分布上限
date = "220623_0.5" 
data_dir = "./data/"

###读取zernike分布下限文件
# path='/home/xianyuer/yuer/21zishiying/21zishiying/1.txt'
# pos = []
# with open(path, 'r') as file_to_read:
#   while True:
#     lines = file_to_read.readline() # 整行读取数据
#     if not lines:
#       break
#       pass
#     line = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
#     pos.append(line)
#     pass
#   bounds = np.array(pos) # 将数据从list类型转换为array类型。
#   pass

###按zernike分布下限生成均匀分布的zernike系数矩阵
'''
nzernike = bounds.shape[1] # 输出列数 
zernike_tot = np.zeros((nbatch*nsnapshots,nzernike)) #50000帧7列
for j in range(nzernike):
    zernike_tot[:,[j]] = np.random.uniform(low = -1*bounds[0][j], high = bounds[0][j], size = (nbatch*nsnapshots, 1))# 生成均匀分布的zernike系数
'''
zernike_tot = np.zeros((nbatch*nsnapshots,nzernike)) #50000帧7列
if not os.path.exists(data_dir):
    os.mkdir(data_dir) # 建立目录
for j in range(2,nzernike):
    zernike_tot[:,[j]] = np.random.uniform(low = low, high = high, size = (nbatch*nsnapshots, 1))# 生成均匀分布的zernike系数
# 均匀分布
#zernike_tot = np.random.uniform(low = low, high = high, size = (nbatch*nsnapshots, nzernike)) # 生成均匀分布的zernike系数
#高斯分布
#zernike_tot = np.random.randn(nbatch*nsnapshots, nzernike) # 生成均匀分布的zernike系数
print("zernike_tot shape = ", np.shape(zernike_tot))
np.save(data_dir+"zernike_%s_%d_%d.npy" %(date, nzernike,nsnapshots), zernike_tot) # 将生成的zernike系数存储在npy文件里
np.savetxt(data_dir+"zernike_%s_%d_%d.dat" %(date, nzernike,nsnapshots), zernike_tot) 