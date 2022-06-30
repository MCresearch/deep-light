#include "gaussian.h"
#include "assert.h"
#ifndef SPOT_H
#define SPOT_H

class Spot
{
    public:
    Spot();
    ~Spot();

    double** value;
    double** fitted_value;
    double quantile;
    int width;
    int window_width;
    int nlocal_max;
    int nlocal_max_ub;
    //double local_max_thre;
    int** local_max_coord;
    Gaussian* gaussian;

    bool read_;
    bool identify_;
    bool fit_;
    bool predict_;


    void readin(ifstream &ifs);
    void identify_local_max();
    void fit_gaussian();
    void predict();
    void clean();
    void calc_quantile();
};

#endif