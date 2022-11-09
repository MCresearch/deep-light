# model_Zernike_loss

## 文件结构
  * [Ch 1: Light](#ch-1-light)
  * [Ch 2: MachineLearning](#ch-2-mechinelearning)
  
---
--- 
## [Ch 1: Light](light/README.md)
光束远场传输模拟程序：本程序用于模拟点光源，在经过由Zernike多项式模拟的相位扰动后得到的远场。程序可以读入随机生成的Zernike多项式系数，且批量生成与之相匹配的远场光强。
  * 程序各个模块的作用；
  * 程序结果展示；
  * 程序运行方法。
  * <font size=3>光束远场传输程序的测试文档，其中包括`32*32`，`64*64`,`128*128`，`256*256`四种网格。需要输入`INPUT.txt`和Zernike系数`.dat`文件。
运行方法：`bash test64.sh`
[具体参数含义](doc/input%20Chinese.md)。</font></b>
---
---
## [Ch 1: MachineLearning](machinelearning/README.md)
基于机器学习的波前相差感知程序：运用机器学习的方法由远场光强学习到波前像差对应的Zernike多项式系数。目前参考的网络有：简化后的Xception模型、Xception模型、ResNet模型。
  * 程序各个模块的作用；
  * 程序结果展示；
  * 程序运行方法。
---
---
