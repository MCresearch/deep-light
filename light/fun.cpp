#include "fun.h"
#define PI 3.141592653589793

void prop1(const int    n_grid,
           const int    n1,
           const double dz,
           const double kp,
           const double aa,
           double*      hr,
           double*      hi)
{
    double tt = 0.0;
    double t0 = 0.0;
    int    j1 = 0;
    tt = dz / (2 * kp) * pow((2 * PI / aa), 2);
    for (int j = 1; j <= n_grid; j++)
    {
        j1 = j - n1;
        t0 = tt * j1 * j1;
        hr[j - 1] = cos(-t0);
        hi[j - 1] = sin(-t0);
    }
}

void evol1(const int n_grid, const double* hr, const double* hi, double** ur, double** ui)
{
    double cx = 0.0;
    for (int j = 1; j <= n_grid; j++)
    {
        for (int i = 1; i <= n_grid; i++)
        {
            cx = ur[i - 1][j - 1];
            ur[i - 1][j - 1] = cx * hr[i - 1] - ui[i - 1][j - 1] * hi[i - 1];
            ui[i - 1][j - 1] = cx * hi[i - 1] + ui[i - 1][j - 1] * hr[i - 1];
        }
    }
    
    for (int j = 1; j <= n_grid; j++)
    {
        for (int i = 1; i <= n_grid; i++)
        {
            cx = ur[i - 1][j - 1];
            ur[i - 1][j - 1] = cx * hr[j - 1] - ui[i - 1][j - 1] * hi[j - 1];
            ui[i - 1][j - 1] = cx * hi[j - 1] + ui[i - 1][j - 1] * hr[j- 1];
        }
    }
}

void mdfph(const int    n_grid,
           const int    n1,
           const double dx,
           const double dta,
           const double ddx,
           const double kp,
           double**     ur,
           double**     ui)
{
    double cx = 0.0;
    double y = 0.0;
    double ey = 0.0;
    double x = 0.0;
    double ex = 0.0;
    double ec = 0.0;
    double c = 0.0;
    double s = 0.0;
    int    i1 = 0;
    int    j1 = 0;

    for (int j = 1; j <= n_grid; j++)
    {
        j1 = j - n1;
        y = (j - n1) * dx;
        ey = kp * y * y * dta / (2.0 * ddx);
        for (int i = 1; i <= n_grid; i++)
        {
            i1 = i - n1;
            x = (i - n1) * dx;
            ex = kp * x * x * dta / (2.0 * ddx);
            ec = ex + ey;
            c = cos(ec);
            s = sin(ec);
            ec = ur[i - 1][j - 1];
            ur[i - 1][j - 1] = ec * c - ui[i - 1][j - 1] * s;
            ui[i - 1][j - 1] = ec * s + ui[i - 1][j - 1] * c;
        }
    }
}

void focusing(const int    n_grid,
              const int    n1,
              const double kp,
              const double dx,
              const double rzf,
              double**     ur,
              double**     ui)
{
    double x = 0.0;
    double y = 0.0;
    double ei = 0.0;
    double c = 0.0;
    double s = 0.0;
    double c0 = 0.0;

    if (abs(rzf) < 1e-10)
        return;
    for (int j = 1; j <= n_grid; j++)
    {
        y = (j - n1) * dx;
        for (int i = 1; i <= n_grid; i++)
        {
            x = (i - n1) * dx;
            ei = -kp * (x * x + y * y) / 2 * rzf;
            c = cos(ei);
            s = sin(ei);
            c0 = ur[i - 1][j - 1];
            ur[i - 1][j - 1] = c0 * c - ui[i - 1][j - 1] * s;
            ui[i - 1][j - 1] = c0 * s + ui[i - 1][j - 1] * c;
        }
    }
}

void output_inIntensity(const int    n_grid,
                        const string path,
                        const int    accuracy,
                        double**     ur,
                        double**     ui)
{
    ofstream outfile1;
    outfile1.open(path, ios::app);
    outfile1.setf(ios::fixed, ios::floatfield);
    outfile1.precision(accuracy);
    if (!outfile1.is_open())
    {
        cout << "open file failure" << endl;
    }
    for (int i = 0; i < n_grid; i++)
    {
        for (int j = 0; j < n_grid; j++)
        {
            outfile1 << pow(ur[i][j], 2) + pow(ui[i][j], 2) << '\t';
        }
        outfile1 << endl;
    }
    outfile1.close();
}

void output_zernike_coeff(const int     n_grid,
                          const string  path,
                          const int     accuracy,
                          const int     maxZnkDim,
                          const double* aznk,
                          const int*    nznk,
                          const double* eznk)
{
    ofstream outfile1;
    outfile1.open(path, ios::app);
    outfile1.setf(ios::fixed, ios::floatfield);
    outfile1.precision(accuracy);
    if (!outfile1.is_open())
    {
        cout << "open file failure" << endl;
    }
    for (int i = 1; i <= maxZnkDim; i++)
    {
        outfile1 << i << "\t" << aznk[i] << '\t' << nznk[i] << "\t" << eznk[i] << endl;
    }
    outfile1.close();
}

void output_inPhase(const int n_grid, const string path, const int accuracy, double** ph)
{
    ofstream outfile1;
    outfile1.open(path, ios::app);
    outfile1.setf(ios::fixed, ios::floatfield);
    outfile1.precision(accuracy);
    if (!outfile1.is_open())
    {
        cout << "open file failure" << endl;
    }
    for (int i = 0; i < n_grid; i++)
    {
        for (int j = 0; j < n_grid; j++)
        {
            outfile1 << ph[i][j] << '\t';
        }
        outfile1 << endl;
    }
    outfile1.close();
}

void output_ur(const int n_grid, const string path, const int accuracy, double** ur)
{
    ofstream outfile1;
    outfile1.open(path, ios::app);
    outfile1.setf(ios::fixed, ios::floatfield);
    outfile1.precision(accuracy);
    if (!outfile1.is_open())
    {
        cout << "open file failure" << endl;
    }
    for (int i = 0; i < n_grid; i++)
    {
        for (int j = 0; j < n_grid; j++)
        {
            outfile1 << ur[i][j] << '\t';
        }
        outfile1 << endl;
    }
    outfile1.close();
}