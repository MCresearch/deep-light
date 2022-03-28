#ifndef INPUT_H
#define INPUT_H
#include <assert.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>

using namespace std;

class Input 
{
public:
    Input();
    ~Input();

    int mm;
    int n_grid;
    int n9;
    int n1;

    int mgs;

    double a0;
    double xx0;
    double aa0;
    double dxy0;

    double plm;
    double zfh;

    double airy;
    double xxz;
    double aaz;
    double dxyz;

    int minZnkDim;
    int maxZnkOrder;

    double rms;
    double eeznk;

    bool INIT(Input &INPUT);
};

//extern Input INPUT;
#endif
