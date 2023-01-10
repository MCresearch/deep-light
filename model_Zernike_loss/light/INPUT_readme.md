# INPUT.txt
本文件用于提供聚焦光束远场传输程序的所需参数。
---
--- 
1. mm：网格数，若使用$x*x$的网格，则$x= 2^{mm}$。
2. mgs：光束截断指数，$pow( ,mgs) = pow( ,truncated-beams)$。
3. a0：初始光场光斑半径/m。
4. xx0：初始光场缓冲区域倍数。
5. plm：波长/m。
6. zfh：传输距离/m。
7. xxz：焦区光场缓冲区倍数。
8. minZnkDim：多项式最小阶数。
9. maxZnkOrder：多项式最大阶次（最大为13）。
10. rms：相位方差。
11. eeznk：多项式系数方差变化指数。
12. Phase_option：
    * 选择 "random" ：生成随机Zernike系数，可不更改aznk_dir。
    * 选择"confirm"：使用确定的Zernike系数，aznk_dir中需要指明文件路径。
13. num_datas：输出帧数。
14. aznk_dir：Zernike系数文件路径，要求该文件内Zernike系数矩阵的size为$(num_datas,maxZnkDim -2)$即，需要忽略Zernike系数的前两阶。
15. dir: 输出文件保存路径。
16. out_inIntensity：是否输出初始远场，输出为“1”，不输出为“0”。
17. out_zernike_coeff：是否输出Zernike系数，输出为“1”，不输出为“0”。
18. out_inPhase：是否输出扰动相位，输出为“1”，不输出为“0”。
19. out_focusing：是否输出聚焦后的光场，输出为“1”，不输出为“0”。
20. out_mdfph1：是否输出坐标正变换后的光场，输出为“1”，不输出为“0”。
21. out_my_fft2d1：是否输出傅里叶变化后倒空间内的光场，输出为“1”，不输出为“0”。
22. out_my_fft2d2：是否输出傅里叶逆变换后的光场，输出为“1”，不输出为“0”。
23. out_mdfph2：是否输出坐标逆变换后的光场，输出为“1”，不输出为“0”。
24.out_outIntensity ：是否输出远场光强，输出为“1”，不输出为“0”。
---
---
