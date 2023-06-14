# model_Zernike_loss

## 文件结构
  * [Ch 1: model_train](#ch-1-model_train)
  * [Ch 2: propagation](#ch-2-propagation)
  * [Ch 3: model_test](#ch-3-model_test)
  * [Ch 4: result_analyse](#ch-4-result_analyse)
  
---
--- 
## Ch 1: model_train
模型训练主程序，其输入文件为`INPUT_model.json`和`INPUT_propagation.json`
其需要调用`propagation.py`和网络`Xception.py`

---
---
## Ch 2: propagation
聚焦光束远场传输程序，输出模型训练所需数据。
其需要调用生成Zernike模式的`Zernike.py`与函数文件`fun.py`。

---
---
## Ch 3: model_test
模型测试。

---
---
## Ch 4: result_analyse
结果分析。

---
---