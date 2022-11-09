# Machine Leraning部分
基于机器学习的波前相差感知程序：运用机器学习的方法由远场光强学习到波前像差对应的Zernike多项式系数。目前参考的网络有：简化后的Xception模型、Xception模型、ResNet模型。

---
---

## 模型原理
Xception[参考文献](https://arxiv.org/abs/1610.02357)，ResNet[参考文献](https://blog.csdn.net/weixin_42206075/article/details/111174874?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-1-111174874-null-null.pc_agg_new_rank&utm_term=keras%E5%AE%9E%E7%8E%B0resnet&spm=1000.2123.3001.4430)。

---
---
## 主要文件介绍
1. xcpetion.py ：Xception网络搭建。
2. xception0919.py：简化后的Xception网络搭建。
3. 35Zernike_model_data_test.py：Xception网络训练。
4. xception_test.py：Xception网络测试。
5. 35resnet.py： ResNet网络训练。
6. resnet_test.py： ResNet网络训练。
7. lossplot.py: 训练中的损失函数可视化。
