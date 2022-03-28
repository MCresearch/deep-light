#ifndef OPTICAL_FIELD
#define OPTICAL_FIELD

#include "Zernike.h"
#include "fft.h"
#include "fun.h"
#include "input.h"


class OPT {
public:
    OPT();
    ~OPT();


    static bool Init_Intensity(Input &INPUT, OPT& opt);
    static bool Init_Phase(Input &INPUT, OPT& opt, const double a1);
    static void numercial_diffraction(Input &INPUT, OPT& opt);

    double** ur;
    double** ui;
    int      maxZnkDim;

    double*  aznk;
    double*  eznk;
    double*  pl;
    double** ph;

    int* nznk;
    int* mznk;
    int* lznk;
};
#endif