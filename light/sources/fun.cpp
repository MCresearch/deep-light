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
    for (int i = 0; i < n_grid; i++)
    {
        for (int j = 0; j < n_grid; j++)
        {
            cx = ur[i][j];
            ur[i][j] = cx * hr[i] - ui[i][j] * hi[i];
            ui[i][j] = cx * hi[i] + ui[i][j] * hr[i];
            cx = ur[i][j];
            ur[i][j] = cx * hr[j] - ui[i][j] * hi[j];
            ui[i][j] = cx * hi[j] + ui[i][j] * hr[j];
        }
    }
/*
    for (int i = 0; i < n_grid; i++)
    {
        for (int j = 0; j < n_grid; j++)
        {
            cx = ur[i][j];
            ur[i][j] = cx * hr[j] - ui[i][j] * hi[j];
            ui[i][j] = cx * hi[j] + ui[i][j] * hr[j];
        }
    }
    */
    
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


    for (int i = 0; i < n_grid; i++)
    {
        x = (i + 1 - n1) * dx;
        ex = kp * x * x * dta / (2.0 * ddx);
        for (int j = 0; j < n_grid; j++)
        {
            y = (j + 1 - n1) * dx;
            ey = kp * y * y * dta / (2.0 * ddx);
            ec = ex + ey;
            c = cos(ec);
            s = sin(ec);
            ec = ur[i][j];
            ur[i][j] = ec * c - ui[i][j] * s;
            ui[i][j] = ec * s + ui[i][j] * c;
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
    for (int i = 0; i < n_grid; i++)
    {
        x = (i + 1 - n1) * dx;
        for (int j = 0; j < n_grid; j++)
        {
            y = (j + 1 - n1) * dx;
            ei = -kp * (x * x + y * y) / 2 * rzf;
            c = cos(ei);
            s = sin(ei);
            c0 = ur[i][j];
            ur[i][j] = c0 * c - ui[i][j] * s;
            ui[i][j] = c0 * s + ui[i][j] * c;
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
    //outfile1 << "#output_inIntensity" << endl;
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
    outfile1 << "#output_zernike_coeff" << endl;
    for (int i = 1; i <= maxZnkDim; i++)
    {
        outfile1 << i << "\t" << aznk[i] << '\t' << nznk[i] << "\t" << eznk[i] << endl;
    }
    outfile1.close();
}

void output_zernike_coeff_0(const int     n_grid,
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
    //outfile1 << "#output_zernike_coeff" << endl;
    for (int i = 1; i <= maxZnkDim; i++)
    {
        outfile1 << aznk[i] << '\t';
    }
    outfile1 << endl;
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
    outfile1 << "#output_inPhase" << endl;
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
    // outfile1 << "#output_ur" << endl;
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

void output_ui(const int n_grid, const string path, const int accuracy, double** ui)
{
    ofstream outfile1;
    outfile1.open(path, ios::app);
    outfile1.setf(ios::fixed, ios::floatfield);
    outfile1.precision(accuracy);
    if (!outfile1.is_open())
    {
        cout << "open file failure" << endl;
    }
    outfile1 << "#output_ui" << endl;
    for (int i = 0; i < n_grid; i++)
    {
        for (int j = 0; j < n_grid; j++)
        {
            outfile1 << ui[i][j] << '\t';
        }
        outfile1 << endl;
    }
    outfile1.close();
}