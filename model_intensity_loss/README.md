# 聚焦光束远场传输数据模拟程序：
  ## gen_data.py 
      为数据模拟主程序，需要调用propagation.py和Zernike.py,读取参数文件INPUT.json。
  #### propagation.py
      包括聚焦光束远场传输所需要的函数
      init_intensity： 初始场生成；
      Zer： Zernike模式及系数生成；
      progagtion： 生成扰动初始场并进行传输最终得到降采样后的远场光强。该函数会用到fun.py中的函数。
  #### Zernike.py
      maxZernike：计算Zernike阶数；
      Zernike： 生成Zernike模式。
  #### INPUT.json
      为输入参数文件。
  ---
  ---
  ## test_propagation
      聚焦光束远场传输数据模拟程序测试文件。
---
---
# 基于机器学习的初始场相位探测模型：
  ## train_model.py
