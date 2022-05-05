#include "Zernike.h"
#define PI 3.141592653589793

void rdm_gauss(double &a1, double &rdmg)
{
    double PI2 = 0.0;
    double asd = 0.0;
    double r01 = 0.0;
    double r02 = 0.0;
    double rq = 0.0;
    PI2 = 2 * PI;
    rq = pow(PI, (8 + PI)) + PI;

    asd = a1;
    asd = asd * rq;
    asd = asd - (int)asd;
    r01 = sqrt(-2 * log(asd));
    asd = asd * rq;
    asd = asd - (int)asd;
    r02 = PI2 * asd;

    rdmg = r01 * cos(r02);
    a1 = asd;
}

int maxZernike(const int nk)
{
    int maxZernike = 0;
    int l = 0;
    int ni = 0;
    int m0 = 0;
    int m1 = 0;
    int m = 0;
    for (ni = 1; ni <= nk; ni++)
    {
        m0 = ni - 2 * (int)(ni / 2);
        m1 = m0 + 2 * (int)(ni / 2);
        for (m = m0; m <= m1; m = m + 2)
        {
            l = l + 1;
            if (m != 0)
                l++;
        }
    }
    return l;
}

void nmlznk(const int maxZnkOrder, int &maxZnkDim, int *&nznk, int *&mznk)
{
    int    maxZernike = 0;
    double fac1 = 0;
    double fac2 = 0;
    double fac3 = 0;
    double fac4 = 0;
    int    l = 0;
    int    ni = 0;
    int    m0 = 0;
    int    m1 = 0;
    int    m = 0;
    int    lz = 0;
    int    j = 0;
    int    nm = 0;

    for (ni = 1; ni <= maxZnkOrder; ni++)
    {
        m0 = ni - 2 * (int)(ni / 2);
        m1 = m0 + 2 * (int)(ni / 2);
        for (m = m0; m <= m1; m = m + 2)
        {
            l = l + 1;
            if (l >= 1)
            {
                nznk[l] = ni;  // zernike系数从1开始储存
                mznk[l] = m;
                lz = lznk_a(l, ni);
                // lznk[l] = lz;
                cout << ni << m << lz << endl;
            }
            if (m != 0)
            {
                l = l + 1;
                if (l >= 1)
                {
                    nznk[l] = ni;  // zernike系数从1开始储存
                    mznk[l] = m;
                    lz = lznk_a(l, ni);
                    // lznk[l] = lz;
                    nm = (ni - m) / 2;
                    cout << ni << m << lz << endl;
                }
            }
            if (l >= maxZnkDim)
            {
                maxZnkDim = l;
                return;
            }
        }
    }
}

void mnznk(const int maxZnkOrder, int &maxZnkDim, int *&nznk, int *&mznk)
{
    int n = 0;
    int m = 0;
    int j = 0;

    for (n = 1; n <= maxZnkOrder; n++)
    {
        for (m = -n; m <= n; m = m + 2)
        {
            j = j + 1;
            nznk[j] = n;
            mznk[j] = m;
            cout << nznk[j] << " " << mznk[j] << " " << endl;
        }
        maxZnkDim = j;
    }
}

void radial_polynomials(const int    N,
                        const double x,
                        const double y,
                        int *&       nznk,
                        int *&       mznk,
                        double *&    zer)
{
    double p = 0.0;
    double c = 0.0;
    if (x == 0)
    {
        cout << "x!=0" << endl;
        exit(0);
    }
    cout << 111 << endl;
    p = sqrt(x * x + y * y);
    c = atan(y / x);
    int i = 0;
    int k = 0;
    int n = 0;
    int m = 0;
    for (i = 0; i < N; i++)
    {
        cout << i << endl;
        n = nznk[i];
        m = mznk[i];
        if (p == 1)
        {
            zer[i] = 1;
        }
        else
        {
            for (k = 0; k <= (n - m) / 2; k++)
            {
                zer[i] = zer[i] + pow(-1,k) * pow(p,n-2*k) * recv(n-k) / recv((n+m)/2-k) / recv((n-m)/2-k);
            }
        }
        if(m >= 0)
        {
            zer[i] = zer[i] * cos(m * c);
        }
        else
        {
            zer[i] = zer[i] * sin(-1 * m * c);
        }
    }
}
// void radial_polynomials(int m, int n,)
int recv(int n)
{
    int sum = 1;

    if (1 == n)
    {
        return 1;
    }

    sum = n * recv(n - 1);

    return sum;
}

int lznk_a(const int l, const int ni)
{
    int lz = 0;
    if ((ni / 2) / 2 != (ni / 2) * 0.5)
    {
        if (l / 2 == l * 0.5)
        {
            lz = 0;
        }
        else
        {
            lz = 1;
        }
    }
    else
    {
        if (l / 2 == l * 0.5)
        {
            lz = 1;
        }
        else
        {
            lz = 0;
        }
    }
    return lz;
}

void zernike_cg(const int N, const double x, const double y, double *&zer)
{
    double r = 0;
    double c2 = 0;
    double s2 = 0;
    double c3 = 0;
    double s3 = 0;
    double c4 = 0;
    double s4 = 0;
    double c5 = 0;
    double s5 = 0;
    double c6 = 0;
    double s6 = 0;
    double c7 = 0;
    double s7 = 0;
    double c8 = 0;
    double s8 = 0;
    double c9 = 0;
    double s9 = 0;
    double c10 = 0;
    double s10 = 0;
    double c11 = 0;
    double s11 = 0;
    double c12 = 0;
    double s12 = 0;
    double c13 = 0;
    double s13 = 0;
    if (N > 104)
    {
        cout << "使用数组越界!" << endl;
        return;
    }
    r = x * x + y * y;
    c2 = x * x - y * y;
    s2 = 2 * x * y;
    for (int ii = 1; ii <= N; ii++)
    {
        if (ii == 1)
            zer[1] = 2.0 * x;
        if (ii == 2)
            zer[2] = 2.0 * y;
        if (ii == 3)
            zer[3] = sqrt(3.0) * (2 * r - 1);
        if (ii == 4)
            zer[4] = sqrt(6.0) * s2;
        if (ii == 5)
            zer[5] = sqrt(6.0) * c2;
        if (ii == 6)
            zer[6] = sqrt(8.0) * y * (3 * r - 2);
        if (ii == 7)
            zer[7] = sqrt(8.0) * x * (3 * r - 2);
        c3 = x * c2 - y * s2;
        s3 = y * c2 + x * s2;
        if (ii == 8)
            zer[8] = sqrt(8.0) * s3;
        if (ii == 9)
            zer[9] = sqrt(8.0) * c3;
        if (ii == 10)
            zer[10] = sqrt(5.0) * (6 * r * r - 6 * r + 1);
        if (ii == 11)
            zer[11] = sqrt(10.0) * (4 * r - 3) * c2;
        if (ii == 12)
            zer[12] = sqrt(10.0) * (4 * r - 3) * s2;
        c4 = c2 * c2 - s2 * s2;
        s4 = 2 * s2 * c2;
        if (ii == 13)
            zer[13] = sqrt(10.0) * c4;
        if (ii == 14)
            zer[14] = sqrt(10.0) * s4;
        if (ii == 15)
            zer[15] = sqrt(12.0) * (10 * r * r - 12 * r + 3) * x;
        if (ii == 16)
            zer[16] = sqrt(12.0) * (10 * r * r - 12 * r + 3) * y;
        if (ii == 17)
            zer[17] = sqrt(12.0) * c3 * (5 * r - 4);
        if (ii == 18)
            zer[18] = sqrt(12.0) * s3 * (5 * r - 4);
        c5 = c2 * c3 - s2 * s3;
        s5 = s2 * c3 + c2 * s3;
        if (ii == 19)
            zer[19] = sqrt(12.0) * c5;
        if (ii == 20)
            zer[20] = sqrt(12.0) * s5;
        if (ii == 21)
            zer[21] = sqrt(7.0) * (20 * r * r * r - 30 * r * r + 12 * r - 1);
        if (ii == 22)
            zer[22] = sqrt(14.0) * (15 * r * r - 20 * r + 6) * s2;
        if (ii == 23)
            zer[23] = sqrt(14.0) * (15 * r * r - 20 * r + 6) * c2;
        if (ii == 24)
            zer[24] = sqrt(14.0) * (6 * r - 5) * s4;
        if (ii == 25)
            zer[25] = sqrt(14.0) * (6 * r - 5) * c4;
        c6 = c3 * c3 - s3 * s3;
        s6 = 2 * s3 * c3;
        if (ii == 26)
            zer[26] = sqrt(14.0) * s6;
        if (ii == 27)
            zer[27] = sqrt(14.0) * c6;
        if (ii == 28)
            zer[28] = 4.0 * (35 * r * r * r - 60 * r * r + 30 * r - 4) * y;
        if (ii == 29)
            zer[29] = 4.0 * (35 * r * r * r - 60 * r * r + 30 * r - 4) * x;
        if (ii == 30)
            zer[30] = 4.0 * (21 * r * r - 30 * r + 10) * s3;
        if (ii == 31)
            zer[31] = 4.0 * (21 * r * r - 30 * r + 10) * c3;
        if (ii == 32)
            zer[32] = 4.0 * (7 * r - 6) * s5;
        if (ii == 33)
            zer[33] = 4.0 * (7 * r - 6) * c5;
        c7 = c4 * c3 - s4 * s3;
        s7 = s4 * c3 + c4 * s3;
        if (ii == 34)
            zer[34] = 4.0 * s7;
        if (ii == 35)
            zer[35] = 4.0 * c7;
        if (ii == 36)
            zer[36] = 3.0 * (70 * r * r * r * r - 140 * r * r * r + 90 * r * r - 20 * r + 1);
        if (ii == 37)
            zer[37] = sqrt(18.0) * (56 * r * r * r - 105 * r * r + 60 * r - 10) * c2;
        if (ii == 38)
            zer[38] = sqrt(18.0) * (56 * r * r * r - 105 * r * r + 60 * r - 10) * s2;
        if (ii == 39)
            zer[39] = sqrt(18.0) * (28 * r * r - 42 * r + 15) * c4;
        if (ii == 40)
            zer[40] = sqrt(18.0) * (28 * r * r - 42 * r + 15) * s4;
        if (ii == 41)
            zer[41] = sqrt(18.0) * (8 * r - 7) * c6;
        if (ii == 42)
            zer[42] = sqrt(18.0) * (8 * r - 7) * s6;
        c8 = c4 * c4 - s4 * s4;
        s8 = 2 * s4 * c4;
        if (ii == 43)
            zer[43] = sqrt(18.0) * c8;
        if (ii == 44)
            zer[44] = sqrt(18.0) * s8;
        if (ii == 45)
            zer[45] =
                sqrt(20.0) * (126 * r * r * r * r - 280 * r * r * r + 210 * r * r - 60 * r + 5) * x;
        if (ii == 46)
            zer[46] =
                sqrt(20.0) * (126 * r * r * r * r - 280 * r * r * r + 210 * r * r - 60 * r + 5) * y;
        if (ii == 47)
            zer[47] = sqrt(20.0) * (84 * r * r * r - 168 * r * r + 105 * r - 20) * c3;
        if (ii == 48)
            zer[48] = sqrt(20.0) * (84 * r * r * r - 168 * r * r + 105 * r - 20) * s3;
        if (ii == 49)
            zer[49] = sqrt(20.0) * (36 * r * r - 56 * r + 21) * c5;
        if (ii == 50)
            zer[50] = sqrt(20.0) * (36 * r * r - 56 * r + 21) * s5;
        if (ii == 51)
            zer[51] = sqrt(20.0) * (9 * r - 8) * c7;
        if (ii == 52)
            zer[52] = sqrt(20.0) * (9 * r - 8) * s7;
        c9 = c5 * c4 - s5 * s4;
        s9 = s5 * c4 + c5 * s4;
        if (ii == 53)
            zer[53] = sqrt(20.0) * c9;
        if (ii == 54)
            zer[54] = sqrt(20.0) * s9;
        if (ii == 55)
            zer[55] = sqrt(11.0) * (252 * pow(r, 5) - 630 * r * r * r * r + 560 * r * r * r -
                                    210 * r * r + 30 * r - 1);
        if (ii == 56)
            zer[56] = sqrt(22.0) *
                      (210 * r * r * r * r - 504 * r * r * r + 420 * r * r - 140 * r + 15) * s2;
        if (ii == 57)
            zer[57] = sqrt(22.0) *
                      (210 * r * r * r * r - 504 * r * r * r + 420 * r * r - 140 * r + 15) * c2;
        if (ii == 58)
            zer[58] = sqrt(22.0) * (120 * r * r * r - 252 * r * r + 168 * r - 35) * s4;
        if (ii == 59)
            zer[59] = sqrt(22.0) * (120 * r * r * r - 252 * r * r + 168 * r - 35) * c4;
        if (ii == 60)
            zer[60] = sqrt(22.0) * (45 * r * r - 72 * r + 28) * s6;
        if (ii == 61)
            zer[61] = sqrt(22.0) * (45 * r * r - 72 * r + 28) * c6;
        if (ii == 62)
            zer[62] = sqrt(22.0) * (10 * r - 9) * s8;
        if (ii == 63)
            zer[63] = sqrt(22.0) * (10 * r - 9) * c8;
        c10 = c5 * c5 - s5 * s5;
        s10 = 2 * s5 * c5;
        if (ii == 64)
            zer[64] = sqrt(22.0) * s10;
        if (ii == 65)
            zer[65] = sqrt(22.0) * c10;
        if (ii == 66)
            zer[66] = sqrt(24.0) *
                      (462 * pow(r, 5) - 1260 * r * r * r * r + 1260 * r * r * r - 560 * r * r +
                       105 * r - 6) *
                      x;
        if (ii == 67)
            zer[67] = sqrt(24.0) *
                      (462 * pow(r, 5) - 1260 * r * r * r * r + 1260 * r * r * r - 560 * r * r +
                       105 * r - 6) *
                      y;
        if (ii == 68)
            zer[68] = sqrt(24.0) *
                      (330 * r * r * r * r - 840 * r * r * r + 756 * r * r - 280 * r + 35) * s3;
        if (ii == 69)
            zer[69] = sqrt(24.) *
                      (330 * r * r * r * r - 840 * r * r * r + 756 * r * r - 280 * r + 35) * c3;
        if (ii == 70)
            zer[70] = sqrt(24.0) * (165 * r * r * r - 360 * r * r + 252 * r - 56) * s5;
        if (ii == 71)
            zer[71] = sqrt(24.0) * (165 * r * r * r - 360 * r * r + 252 * r - 56) * c5;
        if (ii == 72)
            zer[72] = sqrt(24.0) * (55 * r * r - 90 * r + 36) * s7;
        if (ii == 73)
            zer[73] = sqrt(24.0) * (55 * r * r - 90 * r + 36) * c7;
        if (ii == 74)
            zer[74] = sqrt(24.0) * (11 * r - 10) * s9;
        if (ii == 75)
            zer[75] = sqrt(24.0) * (11 * r - 10) * c9;

        c11 = c6 * c5 - s6 * s5;
        s11 = s6 * c5 + c6 * s5;
        if (ii == 76)
            zer[76] = sqrt(24.0) * s11;
        if (ii == 77)
            zer[77] = sqrt(24.0) * c11;

        if (ii == 78)
            zer[78] = sqrt(13.) * (924 * pow(r, 6) - 2772 * pow(r, 5) + 3150 * pow(r, 4) -
                                   1680 * pow(r, 3) + 420 * pow(r, 2) - 42 * r + 1);
        if (ii == 79)
            zer[79] = sqrt(26.) *
                      (792 * pow(r, 5) - 2310 * pow(r, 4) + 2520 * pow(r, 3) - 1260 * pow(r, 2) +
                       280 * r - 21) *
                      c2;
        if (ii == 80)
            zer[80] = sqrt(26.) *
                      (792 * pow(r, 5) - 2310 * pow(r, 4) + 2520 * pow(r, 3) - 1260 * pow(r, 2) +
                       280 * r - 21) *
                      s2;

        if (ii == 81)
            zer[81] = sqrt(26.0) *
                      (495 * r * r * r * r - 1320 * r * r * r + 1260 * r * r - 504 * r + 70) * c4;
        if (ii == 82)
            zer[82] = sqrt(26.0) *
                      (495 * r * r * r * r - 1320 * r * r * r + 1260 * r * r - 504 * r + 70) * s4;
        if (ii == 83)
            zer[83] = sqrt(26.0) * (220 * r * r * r - 495 * r * r + 360 * r - 84) * c6;
        if (ii == 84)
            zer[84] = sqrt(26.0) * (220 * r * r * r - 495 * r * r + 360 * r - 84) * s6;
        if (ii == 85)
            zer[85] = sqrt(26.0) * (66 * r * r - 110 * r + 45) * c8;
        if (ii == 86)
            zer[86] = sqrt(26.0) * (66 * r * r - 110 * r + 45) * s8;
        if (ii == 87)
            zer[87] = sqrt(26.0) * (12 * r - 11) * c10;
        if (ii == 88)
            zer[88] = sqrt(26.0) * (12 * r - 11) * s10;

        c12 = c6 * c6 - s6 * s6;
        s12 = 2 * s6 * c6;
        if (ii == 89)
            zer[89] = sqrt(26.0) * c12;
        if (ii == 90)
            zer[90] = sqrt(26.0) * s12;
        if (ii == 91)
            zer[91] = sqrt(28.0) *
                      (1716 * pow(r, 6) - 5544 * pow(r, 5) + 6930 * r * r * r * r -
                       4200 * r * r * r + 1260 * r * r - 168 * r + 7) *
                      x;
        if (ii == 92)
            zer[92] = sqrt(28.0) *
                      (1716 * pow(r, 6) - 5544 * pow(r, 5) + 6930 * r * r * r * r -
                       4200 * r * r * r + 1260 * r * r - 168 * r + 7) *
                      y;
        if (ii == 93)
            zer[93] = sqrt(28.0) *
                      (1287 * pow(r, 5) - 3960 * r * r * r * r + 4620 * r * r * r - 2520 * r * r +
                       630 * r - 56) *
                      c3;
        if (ii == 94)
            zer[94] = sqrt(28.0) *
                      (1287 * pow(r, 5) - 3960 * r * r * r * r + 4620 * r * r * r - 2520 * r * r +
                       630 * r - 56) *
                      s3;
        if (ii == 95)
            zer[95] = sqrt(28.0) *
                      (715 * r * r * r * r - 1980 * r * r * r + 1980 * r * r - 840 * r + 126) * c5;
        if (ii == 96)
            zer[96] = sqrt(28.0) *
                      (715 * r * r * r * r - 1980 * r * r * r + 1980 * r * r - 840 * r + 126) * s5;
        if (ii == 97)
            zer[97] = sqrt(28.0) * (286 * r * r * r - 660 * r * r + 495 * r - 120) * c7;
        if (ii == 98)
            zer[98] = sqrt(28.0) * (286 * r * r * r - 660 * r * r + 495 * r - 120) * s7;
        if (ii == 99)
            zer[99] = sqrt(28.0) * (78 * r * r - 132 * r + 55) * c9;
        if (ii == 100)
            zer[100] = sqrt(28.0) * (78 * r * r - 132 * r + 55) * s9;
        if (ii == 101)
            zer[101] = sqrt(28.0) * (13 * r - 12) * c11;
        if (ii == 102)
            zer[102] = sqrt(28.0) * (13 * r - 12) * s11;

        c13 = c7 * c6 - s7 * s6;
        s13 = s7 * c6 + c7 * s6;
        if (ii == 103)
            zer[103] = sqrt(28.0) * c13;
        if (ii == 104)
            zer[104] = sqrt(28.0) * s13;
    }
}
