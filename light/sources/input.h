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

class Input {
public:
    Input();
    ~Input();

    int mm;      ///< mm = log_{2}(Number of grid-point)
    int n_grid;  ///< Number of grid-point
    int n9;      ///< Number of grid-point+9
    int n1;      ///< Number of grid-point/ 2 + 1
    int mgs;     ///< Truncated beams

    double a0;    ///< The spot radius of initial fields
    double xx0;   ///< Multiple of initial light field
    double aa0;   ///< aa0 = xx0 * a0
    double dxy0;  ///< dxy0 = aa0 / n_{grid}

    double plm;  ///< Wave length
    double zfh;  ///< Transmission distance

    double airy;  ///< airy = 1.22 * plm * zfh / (2 * a0)
    double xxz;   ///< Multiple of focal light field buffer area
    double aaz;   ///< aaz = INPUT.airy * INPUT.xxz
    double dxyz;  ///< dxyz = aaz / n_grid

    int minZnkDim;    ///< Minimum order of a polynomial
    int maxZnkOrder;  ///< Maximum degree of polynomial(MAX 13).

    double rms;    ///< Phase variance
    double eeznk;  ///< Polynomial coefficient variance change index.

    bool INIT(Input &INPUT);

    string Phase_option;
    string dir;
    string aznk_dir;
    int num_datas;
    /** Read in input parameters
     */
    int out_inIntensity;
    int out_zernike_coeff;
    int out_inPhase;
    int out_focusing;
    int out_mdfph1;
    int out_my_fft2d1;
    int out_evol1;
    int out_my_fft2d2;
    int out_mdfph2;
    int out_outIntensity;
};

// extern Input INPUT;
#endif
