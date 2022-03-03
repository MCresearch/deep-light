#include "FFt.h"
#include <math.h>
#include <cmath>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>

FFT::FFT(){}

FFT::~FFT(){}
bool FFT::fft_initialize(int mm, int n, FFT &fft)
{
    double PI = 3.141592653589793;
    int i = 0;
    int ik = 0;
    int ikk = 0;
    int k = 0;
    int ks = 0;
    int kl = 0;
    int ka = 0;
    int kb = 0;
    int kc = 0;
    double pp = 0.0;

    fft.kk = new int[mm](); //???fortran为mm-1 注意是否正确
	fft.kj = new int[mm](); //???fortran为mm-1 注意是否正确
    fft.km0 = new int[n+1](); //???fortran为n
    fft.km = new int**[n+1](); //???fortran为n
	for(i = 0; i <= n; i++)
	{
		fft.km[i] = new int*[mm]();  //??? fortran km的第二个下标是从1开始的现在为0
        for(int j = 0; j < mm ; j++)
        {
            fft.km[i][j] = new int[3]; //??? fortran km的第三个下标是从1开始的现在为0
        }
	}
    fft.wr = new double[2*n](); //???fortran为-n:n-1 现在变为0:2n-1 注意是否正确
    fft.wi = new double[2*n](); //???fortran为-n:n-1 现在变为0:2n-1 注意是否正确
	fft.mm = mm;
    fft.n = n;

    for(ikk = 0; ikk <= n-1; ikk++)
    {
        itoc(ikk, fft.kk, mm);
        for(k = 0; k <= mm-1; k++)
        {
            fft.kj[k] = fft.kk[mm - k - 1]; 
        }
        fft.km0[ikk] = ctoi(fft.kj, mm); 
    }
    
    for(ks = 1; ks <= mm; ks++)
    {
        for(ikk = 0; ikk <= n-1; ikk++)
        {
            itoc(ikk, fft.kk,mm);
            kl = fft.kk[ks-1];

            fft.kk[ks-1] = 0;
            ka = ctoi(fft.kk, mm);
            fft.kk[ks-1] = 1;
            kb = ctoi(fft.kk, mm);

            fft.kk[ks-1] = kl;
            for(i = 0; i <= mm-1; i++)
            {
                fft.kj[i] = 0;
            }
            for(k = 0; k <= ks-1; k++)
            {
                fft.kj[k+mm-ks] = fft.kk[k];
            }
            kc = ctoi(fft.kj, mm);
            fft.km[ikk][ks-1][0] = ka; // fortran ks 
	        fft.km[ikk][ks-1][1] = kb;
	        fft.km[ikk][ks-1][2] = kc;
        }
    }
    pp = 2 * PI / n;
    for(k = 0; k <= 2*n-1; k++)
    {
       fft.wr[k] = cos((k-n) * pp); //fortran -n n-1 now 0 2n-1 
       fft.wi[k] = sin((k-n) * pp);
    }
	    
}

void FFT::itoc(int ik, int* &kk, int mm)
{
    int m = 0;
    int ikk = 0;
    
    ikk = ik;
    for(m = mm-1; m >= 0; m--)
    {
        kk[m] = ikk / pow(2,m);
        ikk = ikk - kk[m] * pow(2,m);

    }
}

int FFT::ctoi(int* &kk, int mm)
{
    int m = 0;
    int ik = 0;

    for(m = mm-1; m >= 0; m--)
    {
        ik = ik + kk[m] * pow(2,m);
    }
    return ik;
}

void FFT::my_fft2d(FFT &fft, int n9, double** &xr, double** &xi, double dx, int kt)
{
    double ar = 0.0;
    double ai = 0.0;
    double d = 0.0;
    double** temp_cr;
    double** temp_ci;
    int n1 = 0;
    int i1 = 0;
    int i2 = 0;
    int i3 = 0;
    int j1 = 0;
    int j2 = 0;
    int j3 = 0;
    int ks = 0;
    int kf = 0;

    temp_cr = new double*[n9](); //???fortran 1 n9 now 0 n9-1
	temp_ci = new double*[n9]();
    for(int i = 0; i < n9; i++)
	{
		temp_cr[i] = new double[n9]();
        temp_ci[i] = new double[n9]();  
	}
    n1 = fft.n / 2;

    for(int j = 0; j <= n1-1; j++)
    {
        j1 = j + n1;
        for(int i = 0; i <= n1-1; i++)
        {
            i1 = i + n1;
            temp_cr[i][j] = xr[fft.km0[i]][fft.km0[j]];
		    temp_cr[i1][j] =  - xr[fft.km0[i1]][fft.km0[j]];
		    temp_cr[i][j1] =  - xr[fft.km0[i]][fft.km0[j1]];
		    temp_cr[i1][j1] = xr[fft.km0[i1]][fft.km0[j1]];
		    temp_ci[i][j] = xi[fft.km0[i]][fft.km0[j]];
		    temp_ci[i1][j] =  - xi[fft.km0[i1]][fft.km0[j]];
		    temp_ci[i][j1] =  - xi[fft.km0[i]][fft.km0[j1]];
		    temp_ci[i1][j1] = xi[fft.km0[i1]][fft.km0[j1]];
        }
    }
    if(kt >0)
    {
        for(int ms = 1; ms <= fft.mm; ms++)
        {
            for(int j = 0; j <= fft.n-1;j++)
            {
                j3 =  - fft.km[j][ms-1][2];
                for(int i = 0; i <= fft.n-1; i++)
                {
                    i3 =  - fft.km[i][ms-1][2];
                    i1 = fft.km[i][ms-1][0] + 1;
                    i2 = fft.km[i][ms-1][1] + 1;
                    j1 = fft.km[j][ms-1][0] + 1;
                    j2 = fft.km[j][ms-1][1] + 1;
                    ar = temp_cr[i2-1][j1-1] + temp_cr[i2-1][j2-1] * fft.wr[j3-fft.n] - temp_ci[i2-1][j2-1] * fft.wi[j3-fft.n];
                    ai = temp_ci[i2-1][j1-1] + temp_cr[i2-1][j2-1] * fft.wi[j3-fft.n] + temp_ci[i2-1][j2-1] * fft.wr[j3-fft.n];
                    xr[i][j] = temp_cr[i1-1][j1-1] + temp_cr[i1-1][j2-1] * fft.wr[j3-fft.n] - temp_ci[i1-1][j2-1]  * \
                                fft.wi[j3-fft.n] + ar * fft.wr[i3-fft.n] - ai * fft.wi[i3-fft.n];
                    xi[i][j] = temp_ci[i1-1][j1-1] + temp_cr[i1-1][j2-1] * fft.wi[j3-fft.n] + temp_ci[i1-1][j2-1] * \
                                fft.wr[j3-fft.n] + ar * fft.wi[i3-fft.n] + ai * fft.wr[i3-fft.n];                
                }
            }
            for(int j = 1; j <= fft.n; j++)
            {
                for(int i = 1; i <= fft.n; i++)
                {
                    temp_cr[i-1][j-1] = xr[i-1][j-1];
                    temp_ci[i-1][j-1] = xi[i-1][j-1];
                }
            }
        }
    }
    else
    {
        for(int ms = 1; ms <= fft.mm; ms++)
        {
            for(int j = 0; j <= fft.n-1;j++)
            {
                j3 =  fft.km[j][ms-1][2];
                for(int i = 0; i <= fft.n-1; i++)
                {
                    i3 = fft.km[i][ms-1][2];
                    i1 = fft.km[i][ms-1][0] + 1;
                    i2 = fft.km[i][ms-1][1] + 1;
                    j1 = fft.km[j][ms-1][0] + 1;
                    j2 = fft.km[j][ms-1][1] + 1;
                    ar = temp_cr[i2-1][j1-1] + temp_cr[i2-1][j2-1] * fft.wr[j3-fft.n] - temp_ci[i2-1][j2-1] * fft.wi[j3-fft.n];
                    ai = temp_ci[i2-1][j1-1] + temp_cr[i2-1][j2-1] * fft.wi[j3-fft.n] + temp_ci[i2-1][j2-1] * fft.wr[j3-fft.n];
                    xr[i][j] = temp_cr[i1-1][j1-1] + temp_cr[i1-1][j2-1] * fft.wr[j3-fft.n] - temp_ci[i1-1][j2-1]  * \
                                fft.wi[j3-fft.n] + ar * fft.wr[i3-fft.n] - ai * fft.wi[i3-fft.n];
                    xi[i][j] = temp_ci[i1-1][j1-1] + temp_cr[i1-1][j2-1] * fft.wi[j3-fft.n] + temp_ci[i1-1][j2-1] * \
                                fft.wr[j3-fft.n] + ar * fft.wi[i3-fft.n] + ai * fft.wr[i3-fft.n];                
                }
            }
            for(int j = 1; j <= fft.n; j++)
            {
                for(int i = 1; i <= fft.n; i++)
                {
                    temp_cr[i-1][j-1] = xr[i-1][j-1];
                    temp_ci[i-1][j-1] = xi[i-1][j-1];
                }
            }
        }
    }
    kf = 1;
    if(abs(kt) == 1)
    {
        d = 1;
    }
    else
    {
        d = dx * dx;
    }

    for(int j = 1; j <= fft.n; j++)
    {
        kf =  - kf;
        for(int i = 1; i <= fft.n; i++)
        {
            kf =  - kf;
            xr[i-1][j-1] = kf * temp_cr[i-1][j-1] * d;
            xi[i-1][j-1] = kf * temp_ci[i-1][j-1] * d;
        }
    }
    
    for(int i = 0; i < n9; i++)
    {
        delete[] temp_cr[i];
        delete[] temp_ci[i];
    }
	delete[] temp_cr;
    delete[] temp_ci;

}
    