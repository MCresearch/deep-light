# 光束远场传输模拟程序使用方法
所有操作在light目录下进行

## step1： 生成远场

进入`tests`文件夹内相应网格的文件夹，以`64_64`为例，

输入参数：更改`INPUT.txt`文件具体[参考](../doc/documentation.md)；

运行命令：`bash test64.sh`；

输出：相应.dat数据输出在`64_64`文件夹内。

## step2： 将远场光强的.dat文件组装为机器学习所需的.npy文件

进入`tests`文件夹内相应网格的文件夹，以`64_64`为例，

输入参数：
```python
nsnapshots = 10000 # 指定帧数
distribution = "0.5"
nzernike = 35
intensity_dir = "/home/xianyuer/yuer/num_machinelearning/deep-light/tests/64_64/dl_outIntensity.dat"
```
同时需要更改
```python
np.save('/home/xianyuer/yuer/num_mechinelearning/deep-light/machinelearning/0620/data/outIntensity_%d_%s_64_%d'%(nzernike,distribution,nsnapshots),intensity)
```
运行命令： `python data.py `

输出：用于机器学习的.npy光强文件。
