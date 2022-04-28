# documentation
## input parameters
`mm`:   $mm = log_{2}(n_{grid}))$.

`n_grid`: Number of grid-point. $n_{grid}= 2^{mm}$.

`n1` : Half grid number. ${n1} = {n_{grid}} / 2 + 1$.

`mgs` :  Truncated beams. pow( , mgs) = pow( ,truncated beams). 

`a0` : The spot radius of initial fields.

`xx0` : Multiple of initial light field buffer area. 

`aa0` : $aa0 = xx0 * a0$.

`dxy0` : $dxy0 = aa0 / n_{grid}$.

`plm` : Wave length.

`zfh` : Transmission distance.

`airy` : $airy = 1.22 * plm * zfh / (2 * a0)$.

`xxz` : Multiple of focal light field buffer area.

`aaz` : $aaz = airy * xxz$.
        

`dxyz` : $dxyz = aaz / n_grid$.

`minZnkDim` : Minimum order of a polynomial.

`maxZnkOrder` : Maximum degree of polynomial(MAX 13).

`rms` : Phase variance.

`eeznk` : Polynomial coefficient variance change index. 

`Phase_option`: "random" phase or "confirm" determinate phase can be selected during phase initialization. 

`dir`: path.

`out_inIntensity`: "1" is output inIntensity, "0" is not output.

`out_zernike_coeff`: "1" is output zernike_coeff, "0" is not output.

`out_inPhase`: "1" is output inPhase, "0" is not output.

`out_focusing`: "1" is output results after focus, "0" is not output.

`out_mdfph1`: "1" is output results after mdfph1, "0" is not output.

`out_my_fft2d1`: "1" is output results after my_fft2d1, "0" is not output.

`out_evol1`: "1" is output results after evol1, "0" is not output.

`out_my_fft2d2`: "1" is output results after my_fft2d2, "0" is not output.

`out_mdfph2`: "1" is output results after mdfph2, "0" is not output.

`out_outIntensity`: "1" is output outIntensity, "0" is not output.






    



