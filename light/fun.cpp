#include "fun.h"
void prop1(int n_grid, int n1, double* &hr, double* &hi, double dz, double kp, double aa)
{
    double tt = 0.0;
    double t0 = 0.0;
	int j1 = 0;
    double PI = 3.141592653589793;
    tt = dz / (2 * kp) * pow((2 * PI / aa) ,2);
	for(int j = 0; j < n_grid; j++)
    {
        j1 = j + 1 - n1;
		t0 = tt * j1 * j1;
		hr[j] = cos(-t0);
		hi[j] = sin(-t0);
    }
}

void evol1(int n_grid,double* &hr, double* &hi, double** &ur, double** &ui)
{
    double cx = 0.0;
    for(int j = 0; j < n_grid; j++)
    {
        for(int i = 0; i < n_grid; i++)
        {
            cx = ur[i][j];
			ur[i][j] = cx * hr[i] - ui[i][j] * hi[i];
			ui[i][j] = cx * hi[i] + ui[i][j] * hr[i];
        }
    }

    for(int j = 0; j < n_grid; j++)
    {
        for(int i = 0; i < n_grid; i++)
        {
            cx = ur[i][j];
			ur[i][j] = cx * hr[i] - ui[i][j] * hi[i];
			ui[i][j] = cx * hi[i] + ui[i][j] * hr[i];
        }
    }
}

void mdfph(int n_grid, int n1, double** &ur, double** &ui, double dx, double dta, double ddx, double kp)
{
    double cx = 0.0;
    double y = 0.0;
    double ey = 0.0;
    double x = 0.0;
    double ex = 0.0;
    double ec = 0.0;
    double c = 0.0;
    double s = 0.0;
    int i1 = 0;
    int j1 = 0;

    for(int j = 0; j < n_grid; j++)
    {
        j1 = j + 1  - n1;
		y = (j - n1) * dx;
		ey = kp * y * y * dta / (2.0 * ddx);
        for(int i = 0; i < n_grid; i++)
        {
            i1 = i + 1  - n1;
			x = (i - n1) * dx;
			ex = kp * x * x * dta / (2.0 * ddx);
			ec = ex + ey;
			c = cos(ec);
			s = sin(ec);
			ec = ur[i][j];
			ur[i][j] = ec * c - ui[i][j] * s;
			ui[i][j] = ec * s + ui[i][j] * c;
        }
    }  
}

void focusing(int n_grid, int n1, double** &ur, double** &ui, double kp, double dx, double rzf)
{
    double x = 0.0;
    double y = 0.0;
    double ei = 0.0;
    double c = 0.0;
    double s = 0.0;
    double c0 = 0.0;

    if(abs(rzf) < 1e-10) return;
    for(int j = 0; j < n_grid; j++)
    {
        y = (j + 1  - n1) * dx;
        for(int i = 0; i < n_grid; i++)
        {
            x = (i + 1  - n1) * dx;
			ei =  - kp * (x * x + y * y) / 2 * rzf;
			c = cos(ei);
			s = sin(ei);
			c0 = ur[i][j];
			ur[i][j] = c0 * c - ui[i][j] * s;
			ui[i][j] = c0 * s + ui[i][j] * c;
        }
    }  


}