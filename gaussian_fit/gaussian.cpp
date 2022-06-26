#include "gaussian.h"

Gaussian::Gaussian()
{

}

Gaussian::~Gaussian(){}

double Gaussian::eval(const double &x, const double &y)
{
    double x12 = pow((x-x0)*cos(alpha)+(y-y0)*sin(alpha), 2)/sigmax2;
    double y12 = pow(-(x-x0)*sin(alpha)+(y-y0)*cos(alpha), 2)/sigmay2;
    return I*exp(-x12-y12);
}

void Gaussian::fit(double** window, const int &width)
{
    int l = width/2;
    double ** log_window = new double*[width];
    double fx2 = 0;
    double fy2 = 0;
    double fxy = 0;
    double fx = 0;
    double fy = 0;
    double f = 0;
    double p = 0;
    double q = 0;
    for (int iy=0; iy<width; iy++) 
    {
        log_window[iy] = new double[width];
        for (int ix=0; ix<width; ix++) 
        {
            double tmp = log(window[iy][ix]);
            log_window[iy][ix] = tmp;
            fx2 += -tmp * pow(ix - l, 2);
            fy2 += -tmp * pow(iy - l, 2);
            fxy += -tmp * (ix - l) * (iy - l);
            fx += -tmp * (ix - l);
            fy += -tmp * (iy - l);
            f -= tmp;
        }
    }
    for (int i=1; i<l+1; i++) 
    {
        p += pow(i, 4);
        q += pow(i, 2);
    }
    p *= 2*width;
    double r = q*2*width;
    q = 4*pow(q, 2);
    double s = pow(width, 2);
    double A = (fx2*s - f*r)/(p*s - pow(r, 2));
    double B = (fx2 - fy2 - (p-q)*A)/(q-p);
    double C = fx/r;
    double D = fy/r;
    double E = (f-r*(A+B))/s;
    double Z = fxy/q;
    if (A != B)
    {
        alpha = atan(Z/(A-B))/2;
        sigmax2 = 2/((A-B)/cos(2*alpha) + A + B);
        sigmay2 = 2/((B-A)/cos(2*alpha) + A + B);
    }
    else if (abs(Z) < 1e-9)
    {
        sigmax2 = 1/A;
        sigmay2 = 1/A;
        alpha = 0;
    }
    else
    {
        alpha = PI/4;
        sigmax2 = 2/(A+B+Z);
        sigmay2 = 2/(A+B-Z);
    }
    double x1 = -sigmax2 * (C*cos(alpha) + D*sin(alpha))/2;
    double y1 = sigmay2 * (C*sin(alpha) - D*cos(alpha))/2;

    x0 = x1 * cos(alpha) - y1 * sin(alpha);
    y0 = x1 * sin(alpha) + y1 * cos(alpha);

    I = exp(pow(x1, 2) / sigmax2 + pow(y1, 2) / sigmay2 - E);

    if (alpha < 0)
    {
        alpha += PI/2;
        double tmp = sigmax2;
        sigmax2 = sigmay2;
        sigmay2 = tmp;
    }
    return;
}

void Gaussian::gaussian_window(double** window, const int &width)
{
    int l = int(width/2);
    for (int iy=0; iy<width; iy++)
    {
        double y = double(iy - l);
        for (int ix=0; ix<width; ix++)
        {
            double x = double(ix - l);
            window[iy][ix] = this->eval(x, y);
        }
    }
    return;
}