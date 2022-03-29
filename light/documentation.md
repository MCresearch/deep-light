# documentation
## input parameters
`mm`:   $mm = log_{2}(n_{grid})$.

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

`eeznk` : Polynomial coefficient variance change index. !'多项式系数方差变化指数




    



