## 文件说明

文件按用途可分为三种类型：功能文件、训练/测试文件、输入文件。
1. 功能文件：
   * fun.py -- 计算光线传输、FFT变换的相关函数
   * Zernike.py -- 计算泽尼克多项式各阶形式、泽尼克多项式阶数和种类数对应关系的相关函数
   * propagation_speed.py -- 计算泽尼克多项式取值、生成泽尼克系数、计算光强等相关函数
   * Xception_model.py -- Xception模型
2. 训练/测试文件：
   * XceptionTrain.py -- 实时生成数据，在给定输入下获得学习模型
   * ModelTest.py -- 给出指定模型的各项测试结果
3. 输入文件：
   * INPUT.json -- 整个文件的所有可调参数输入

## 使用方法

1. 【训练】将INPUT.json文件“Train”键值下的相关参数调成所需值，运行XceptionTrain.py得到模型文件，文件存储在同目录的model文件夹下，命名格式为“训练日期_泽尼克阶数_网格尺寸_Epoch_BatchSize_相位方差_学习率(atob:1e-a~1e-b)”。
2. 【模型测试】将INPUT.json文件“Test”键值下的相关参数调成所需值，运行XceptionTrain.py得到测试结果，文件存储在同目录中自主命名的文件夹下（INPUT.json中设置）。

## 注意：除输入文件外，其余文件皆不需要修改！！！