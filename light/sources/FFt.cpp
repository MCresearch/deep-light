#include "FFt.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <math.h>
#include <sstream>
#include <string>

#define PI 3.141592653589793

FFT::FFT() {}

FFT::~FFT() {}
bool FFT::fft_initialize(const int mm, const int n, FFT& fft)
{
    int    i = 0;
    int    ik = 0;
    int    ikk = 0;
    int    k = 0;
    int    ks = 0;
    int    kl = 0;
    int    ka = 0;
    int    kb = 0;
    int    kc = 0;
    double pp = 0.0;

    fft.kk = new int[mm]();       //???fortran is mm-1 注意是否正确
    fft.kj = new int[mm]();       //???fortran is mm-1 注意是否正确
    fft.km0 = new int[n + 1]();   //???fortran is n
    fft.km = new int**[n + 1]();  //???fortran is n

    for (i = 0; i <= n; i++)
    {
        fft.km[i] = new int*[mm]();  //??? fortran km的第二个下标是从1开始的现在为0
        for (int j = 0; j < mm; j++)
        {
            fft.km[i][j] = new int[3];  //??? fortran km的第三个下标是从1开始的现在为0
        }
    }
    fft.wr = new double[2 * n]();  //???fortran为-n:n-1 现在变为0:2n-1 注意是否正确
    fft.wi = new double[2 * n]();  //???fortran为-n:n-1 现在变为0:2n-1 注意是否正确
    fft.mm = mm;
    fft.n = n;

    /*
    ofstream outfile1111;
    ofstream outfile1121;
    outfile1111.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_kk0.dat", ios::app);
    outfile1121.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_kj0.dat", ios::app);
    outfile1111.setf(ios::fixed, ios::floatfield);
    outfile1111.precision(6);
    outfile1121.setf(ios::fixed, ios::floatfield);
    outfile1121.precision(6);
    */
    for (ikk = 0; ikk <= n - 1; ikk++)
    {
        itoc(ikk, mm, fft.kk);

        for (k = 0; k <= mm - 1; k++)
        {
            fft.kj[k] = fft.kk[mm - k - 1];
        }
        //**********************************************
        for (int i = 0; i <= mm - 1; i++)
        {
            //outfile1111 << ikk << "\t" << fft.kk[i] << endl;
            //outfile1121 << ikk << "\t" << fft.kj[i] << endl;
        }
        //**********************************************
        fft.km0[ikk] = ctoi(fft.kj, mm);
        // cout <<"ikk" << " " << fft.km0[ikk] << endl;
    }

    //**********************************************
    //outfile1111.close();
    //outfile1121.close();
    //***********************************************

    for (ks = 1; ks <= mm; ks++)
    {
        for (ikk = 0; ikk <= n - 1; ikk++)
        {
            itoc(ikk, mm, fft.kk);

            kl = fft.kk[ks - 1];
            if (ks == 1)
            {
                // cout << "kl" << "\t" << ikk << "\t" << kl << endl;
            }
            fft.kk[ks - 1] = 0;
            ka = ctoi(fft.kk, mm);
            fft.kk[ks - 1] = 1;
            kb = ctoi(fft.kk, mm);

            fft.kk[ks - 1] = kl;
            for (i = 0; i <= mm - 1; i++)
            {
                fft.kj[i] = 0;
            }
            for (k = 0; k <= ks - 1; k++)
            {
                fft.kj[k + mm - ks] = fft.kk[k];
            }
            kc = ctoi(fft.kj, mm);
            fft.km[ikk][ks - 1][0] = ka;  // fortran ks
            fft.km[ikk][ks - 1][1] = kb;
            fft.km[ikk][ks - 1][2] = kc;
        }
    }
    pp = 2 * PI / n;
    for (k = 0; k <= 2 * n - 1; k++)
    {
        fft.wr[k] = cos((k - n) * pp);  // fortran -n n-1 now 0 2n-1
        fft.wi[k] = sin((k - n) * pp);
    }

    /*
    ofstream outfile111;
    ofstream outfile112;
    outfile111.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_kk.dat", ios::app);
    outfile112.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_kj.dat", ios::app);
    outfile111.setf(ios::fixed, ios::floatfield);
    outfile111.precision(6);
    outfile112.setf(ios::fixed, ios::floatfield);
    outfile112.precision(6);

    for (int i = 0; i <= mm - 1; i++)
    {
        outfile111 << fft.kk[i] << endl;
        outfile112 << fft.kj[i] << endl;
    }
    outfile111.close();
    outfile112.close();
    */
   /*
    ofstream outfile113;
    outfile113.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_km0.dat", ios::app);
    outfile113.setf(ios::fixed, ios::floatfield);
    outfile113.precision(6);

    for (int i = 0; i <= n; i++)
    {
        outfile113 << fft.km0[i] << endl;
    }
    outfile113.close();
    //*********************************************
    ofstream outfile114;
    ofstream outfile115;
    outfile114.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_wr.dat", ios::app);
    outfile115.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_wi.dat", ios::app);
    outfile114.setf(ios::fixed, ios::floatfield);
    outfile114.precision(6);
    outfile115.setf(ios::fixed, ios::floatfield);
    outfile115.precision(6);
    for (int i = 0; i <= 2 * n - 1; i++)
    {
        outfile114 << fft.wr[i] << endl;
        outfile115 << fft.wi[i] << endl;
    }
    outfile114.close();
    outfile115.close();
    //*********************************************
    ofstream outfile116;
    ofstream outfile117;
    ofstream outfile118;
    outfile116.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_km_0.dat", ios::app);
    outfile117.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_km_1.dat", ios::app);
    outfile118.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_km_2.dat", ios::app);

    outfile116.setf(ios::fixed, ios::floatfield);
    outfile117.setf(ios::fixed, ios::floatfield);
    outfile118.setf(ios::fixed, ios::floatfield);

    outfile116.precision(6);
    outfile117.precision(6);
    outfile118.precision(6);

    for (int i = 0; i <= n - 1; i++)
    {
        for (int j = 0; j <= mm - 1; j++)
        {
            outfile116 << fft.km[i][j][0] << "\t";
            outfile117 << fft.km[i][j][1] << "\t";
            outfile118 << fft.km[i][j][2] << "\t";
        }
        outfile116 << endl;
        outfile117 << endl;
        outfile118 << endl;
    }
    outfile116.close();
    outfile117.close();
    outfile118.close();
    // cout << "km(100,4,3)" << fft.km[58][4][0] << endl;
    */
}

void FFT::itoc(const int ik, const int mm, int* kk)
{
    int m = 0;
    int ikk = 0;

    ikk = ik;
    for (m = mm - 1; m >= 0; m--)
    {
        kk[m] = ikk / pow(2, m);
        ikk = ikk - kk[m] * pow(2, m);
    }
}

int FFT::ctoi(const int* kk, const int mm)
{
    int m = 0;
    int ik = 0;

    for (m = mm - 1; m >= 0; m--)
    {
        ik = ik + kk[m] * pow(2, m);
    }
    return ik;
}

void FFT::my_fft2d(const FFT&   fft,
                   const int    n9,
                   const double dx,
                   const int    kt,
                   double**     xr,
                   double**     xi)
{
    double   ar = 0.0;
    double   ai = 0.0;
    double   d = 0.0;
    double** temp_cr;
    double** temp_ci;
    int      n1 = 0;
    int      i1 = 0;
    int      i2 = 0;
    int      i3 = 0;
    int      j1 = 0;
    int      j2 = 0;
    int      j3 = 0;
    int      ks = 0;
    int      kf = 0;
    int      n20 = 0;
    int      n21 = 0;
    int      m20 = 0;
    int      m21 = 0;


    temp_cr = new double*[n9]();  //???fortran 1 n9 now 0 n9-1
    temp_ci = new double*[n9]();

    for (int i = 0; i < n9; i++)
    {
        temp_cr[i] = new double[n9]();
        temp_ci[i] = new double[n9]();
    }
    for (int i = 0; i < n9; i++)
    {
        for (int j = 0; j < n9; j++)
        {
            temp_cr[i][j] = 0;
            temp_ci[i][j] = 0;
        }
    }
    n1 = fft.n / 2;

    for (int j = 0; j <= n1 - 1; j++)
    {
        j1 = j + n1;
        for (int i = 0; i <= n1 - 1; i++)
        {
            i1 = i + n1;

            m20 = fft.km0[i];
            n20 = fft.km0[j];
            m21 = fft.km0[i1];
            n21 = fft.km0[j1];

            temp_cr[i][j] = xr[m20][n20];
            temp_cr[i1][j] = -xr[m21][n20];
            temp_cr[i][j1] = -xr[m20][n21];
            temp_cr[i1][j1] = xr[m21][n21];

            temp_ci[i][j] = xi[m20][n20];
            temp_ci[i1][j] = -xi[m21][n20];
            temp_ci[i][j1] = -xi[m20][n21];
            temp_ci[i1][j1] = xi[m21][n21];
        }
    }
/*
    ofstream outfile121;
    ofstream outfile122;
    ofstream outfile123;
    ofstream outfile124;
    outfile121.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_temp_cr1.dat", ios::ate);
    outfile122.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_temp_ci1.dat", ios::ate);
    outfile123.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_temp_xr1.dat", ios::ate);
    outfile124.open("/home/xianyuer/yuer/num/tests/fft/dl_fft_in_temp_xi1.dat", ios::ate);

    outfile121.setf(ios::fixed, ios::floatfield);
    outfile121.precision(6);
    outfile122.setf(ios::fixed, ios::floatfield);
    outfile122.precision(6);
    outfile123.setf(ios::fixed, ios::floatfield);
    outfile123.precision(6);
    outfile124.setf(ios::fixed, ios::floatfield);
    outfile124.precision(6);

    for (int i = 0; i < n9; i++)
    {
        for (int j = 0; j < n9; j++)
        {
            outfile121 << temp_cr[i][j] << "\t";
            outfile122 << temp_ci[i][j] << "\t";
            outfile123 << xr[i][j] << "\t";
            outfile124 << xi[i][j] << "\t";
        }
        outfile121 << endl;
        outfile122 << endl;
        outfile123 << endl;
        outfile124 << endl;
    }
    outfile121.close();
    outfile122.close();
    outfile123.close();
    outfile124.close();
*/

    //*********************************************
    if (kt > 0)
    {
        for (int ms = 1; ms <= fft.mm; ms++)
        {
            for (int j = 0; j <= fft.n - 1; j++)
            {
                j3 = -fft.km[j][ms - 1][2];
                for (int i = 0; i <= fft.n - 1; i++)
                {
                    i3 = -fft.km[i][ms - 1][2];
                    i1 = fft.km[i][ms - 1][0] + 1;
                    i2 = fft.km[i][ms - 1][1] + 1;
                    j1 = fft.km[j][ms - 1][0] + 1;
                    j2 = fft.km[j][ms - 1][1] + 1;
                    ar = temp_cr[i2 - 1][j1 - 1] + temp_cr[i2 - 1][j2 - 1] * fft.wr[j3 + fft.n] -
                         temp_ci[i2 - 1][j2 - 1] * fft.wi[j3 + fft.n];
                    ai = temp_ci[i2 - 1][j1 - 1] + temp_cr[i2 - 1][j2 - 1] * fft.wi[j3 + fft.n] +
                         temp_ci[i2 - 1][j2 - 1] * fft.wr[j3 + fft.n];
                    xr[i][j] = temp_cr[i1 - 1][j1 - 1] +
                               temp_cr[i1 - 1][j2 - 1] * fft.wr[j3 + fft.n] -
                               temp_ci[i1 - 1][j2 - 1] * fft.wi[j3 + fft.n] +
                               ar * fft.wr[i3 + fft.n] - ai * fft.wi[i3 + fft.n];
                    xi[i][j] = temp_ci[i1 - 1][j1 - 1] +
                               temp_cr[i1 - 1][j2 - 1] * fft.wi[j3 + fft.n] +
                               temp_ci[i1 - 1][j2 - 1] * fft.wr[j3 + fft.n] +
                               ar * fft.wi[i3 + fft.n] + ai * fft.wr[i3 + fft.n];
                }
            }
            for (int j = 1; j <= fft.n; j++)
            {
                for (int i = 1; i <= fft.n; i++)
                {
                    temp_cr[i - 1][j - 1] = xr[i - 1][j - 1];
                    temp_ci[i - 1][j - 1] = xi[i - 1][j - 1];
                }
            }
        }
    }
    else
    {
        for (int ms = 1; ms <= fft.mm; ms++)
        {
            for (int j = 0; j <= fft.n - 1; j++)
            {
                j3 = fft.km[j][ms - 1][2];
                for (int i = 0; i <= fft.n - 1; i++)
                {
                    i3 = fft.km[i][ms - 1][2];
                    i1 = fft.km[i][ms - 1][0] + 1;
                    i2 = fft.km[i][ms - 1][1] + 1;
                    j1 = fft.km[j][ms - 1][0] + 1;
                    j2 = fft.km[j][ms - 1][1] + 1;
                    ar = temp_cr[i2 - 1][j1 - 1] + temp_cr[i2 - 1][j2 - 1] * fft.wr[j3 + fft.n] -
                         temp_ci[i2 - 1][j2 - 1] * fft.wi[j3 + fft.n];
                    ai = temp_ci[i2 - 1][j1 - 1] + temp_cr[i2 - 1][j2 - 1] * fft.wi[j3 + fft.n] +
                         temp_ci[i2 - 1][j2 - 1] * fft.wr[j3 + fft.n];
                    xr[i][j] = temp_cr[i1 - 1][j1 - 1] +
                               temp_cr[i1 - 1][j2 - 1] * fft.wr[j3 + fft.n] -
                               temp_ci[i1 - 1][j2 - 1] * fft.wi[j3 + fft.n] +
                               ar * fft.wr[i3 + fft.n] - ai * fft.wi[i3 + fft.n];
                    xi[i][j] = temp_ci[i1 - 1][j1 - 1] +
                               temp_cr[i1 - 1][j2 - 1] * fft.wi[j3 + fft.n] +
                               temp_ci[i1 - 1][j2 - 1] * fft.wr[j3 + fft.n] +
                               ar * fft.wi[i3 + fft.n] + ai * fft.wr[i3 + fft.n];
                }
            }
            for (int j = 1; j <= fft.n; j++)
            {
                for (int i = 1; i <= fft.n; i++)
                {
                    temp_cr[i - 1][j - 1] = xr[i - 1][j - 1];
                    temp_ci[i - 1][j - 1] = xi[i - 1][j - 1];
                }
            }
        }
    }
    kf = 1;
    if (abs(kt) == 1)
    {
        d = 1;
    }
    else
    {
        d = dx * dx;
    }

    for (int j = 1; j <= fft.n; j++)
    {
        kf = -kf;
        for (int i = 1; i <= fft.n; i++)
        {
            kf = -kf;
            xr[i - 1][j - 1] = kf * temp_cr[i - 1][j - 1] * d;
            xi[i - 1][j - 1] = kf * temp_ci[i - 1][j - 1] * d;
        }
    }

    for (int i = 0; i < n9; i++)
    {
        delete[] temp_cr[i];
        delete[] temp_ci[i];
    }
    delete[] temp_cr;
    delete[] temp_ci;

    
}
