# Deep-Light
## 仓库说明

本仓库致力于运用机器学习的方法研究由远场光强预测初始场扰动相位。本仓库一共有两种模型，分别为以Zernike系数为损失函数的相位探测模型以及以远场光强为损失函数的相位探测模型。未来这两种模型会合并为统一的框架。

## 仓库结构

  * [Ch 1: Doc](#ch-1-doc)
  * [Ch 2: model_Zernike_loss](#ch-2-model_Zernike_loss)
  * [Ch 3: model_intensity_loss](#ch-3-model_intensity_loss)
  
---
---

## [Ch 1: Doc](doc/README.md)
本目录主要介绍了程序所用的算法以及必要的程序说明文件。
 1. [光束远场传输程序输入参数说明](doc/input%20Chinese.md)
  * 介绍了光束远场传输模拟程序中所有输入参数的具体含义以及选择范围。
 2. [基于机器学习的波前相差感知程序说明]()  
  * 介绍了程序所选取的神经网络以及相对应的输入参数。
---
---  
## [Ch 2: model_Zernike_loss](model_Zernike_loss/README.md)
基于机器学习的波前相差感知程序：运用机器学习的方法由远场光强学习到波前像差对应的Zernike多项式系数，其中损失函数为Zernike系数的均方误差。目前参考的网络有：简化后的Xception模型、Xception模型、ResNet模型。
  * 程序各个模块的作用；
  * 程序结果展示；
  * 程序运行方法。
 
---
光束远场传输模拟程序：用c++语言，搭建了用于模拟点光源，在经过由Zernike多项式模拟的相位扰动后得到的远场的程序。程序可以读入随机生成的Zernike多项式系数，且批量生成与之相匹配的远场光强。
  * 程序各个模块的作用；
  * 程序结果展示；
  * 程序运行方法。
  * <font size=3>光束远场传输程序的测试文档，其中包括`32*32`，`64*64`,`128*128`，`256*256`四种网格。需要输入`INPUT.txt`和Zernike系数`.dat`文件。
运行方法：`bash test64.sh`
[具体参数含义](doc/input%20Chinese.md)。</font></b>
---
---
## [Ch 3: model_intensity_loss](model_intensity_loss/README.md)
基于机器学习的波前相差感知程序：运用机器学习的方法由远场光强学习到波前像差对应的Zernike多项式系数，其中损失函数为远场光强的均方误差。
光束远场传输模拟程序：用python语言，搭建了用于模拟点光源，在经过由Zernike多项式模拟的相位扰动后得到的远场的程序。

---
---
