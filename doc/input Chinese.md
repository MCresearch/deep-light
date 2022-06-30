# documentation
## input parameters
`mm`:   $mm = log_{2}(n_{grid}))$。

`n_grid`: 网格数。$n_{grid}= 2^{mm}$。

`n1` : 半网格数。 ${n1} = {n_{grid}} / 2 + 1$。

`mgs` : 光束截断指数。 pow( , mgs) = pow( ,truncated beams)。

`a0` : 初始光场光斑半径/m。

`xx0` : 初始光场缓冲区倍数。 

`aa0` : 初始光场计算区域/m$aa0 = xx0 * a0$.

`dxy0` : 初始光场网格尺寸/m。$dxy0 = aa0 / n_{grid}$。

`plm` : 波长/m。

`zfh` : 传输距离/m。

`airy` : 理想衍射光斑半径/m。$airy = 1.22 * plm * zfh / (2 * a0)$。

`xxz` : 焦区光场缓冲区倍数。

`aaz` : 焦区光场计算区域/m。$aaz = airy * xxz$。
        

`dxyz` : 焦区光场网格尺寸/m。$dxyz = aaz / n_grid$。

`minZnkDim` : 多项式最小阶数。

`maxZnkOrder` : 多项式最大阶次（最大为13）。

`rms` : 相位方差。

`eeznk` : 多项式系数方差变化指数。

`Phase_option`: 选择随机相位或者固定相位。"random" 为随机相位。"confirm"为固定相位。 

`dir`: 路径。

`aznk_dir`: Zernike系数矩阵文件路径(.dat)。

`num_datas`: 帧数。 

`out_inIntensity`: "1" 为输出初始光强, "0" 为不输出。

`out_zernike_coeff`: "1" 为输出Zernike系数, "0" 为不输出。

`out_inPhase`: "1" 为输出加入扰动相位后的光场, "0" 为不输出。

`out_focusing`: "1" 为输出聚焦后的光场, "0" 为不输出。

`out_mdfph1`: "1" 为输出做完第一次坐标变换后的光场, "0" 为不输出。

`out_my_fft2d1`: "1" 为输出作完快速傅里叶变换后的光场, "0" 为不输出。

`out_evol1`: "1" 为输出在倒空间内完成光束传输后的光场, "0" 为不输出。

`out_my_fft2d2`: "1" 为输出做完快速傅里叶逆变换的光场, "0" 为不输出。

`out_mdfph2`: "1" 为输出坐标逆变换后的光场, "0" 为不输出。

`out_outIntensity`: "1" 为输出远场光强, "0" 为不输出。






    



