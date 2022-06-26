#include <iostream>
#include <fstream>
#include "input.h"
#include <math.h>
#include <cmath>
#ifndef GAUSSIAN_H
#define GAUSSIAN_H
#define PI 3.1415926535
class Gaussian
{
    public:
    Gaussian();
    ~Gaussian();

    // parameters for Gaussian function
    double I;
    double x0, y0;
    double sigmax2, sigmay2;
    double alpha;

    double eval(const double &x, const double &y);
    void fit(double** window, const int &width);
    void gaussian_window(double** window, const int &width);

};

#endif