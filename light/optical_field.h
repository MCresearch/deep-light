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


    static bool Init_Intensity(Input& INPUT, OPT& opt);
    static bool Init_Phase(Input& INPUT, OPT& opt, const double a1, const string type);
    static void numercial_diffraction(Input& INPUT, OPT& opt);

    double** ur;         ///< The real part of the transformation
    double** ui;         ///< The imaginary part of the transformation
    int      maxZnkDim;  ///< Maximum order of a polynomial.

    double*  aznk;
    double*  eznk;
    double*  pl;  ///< cg Zernike coefficient
    double** ph;  ///< phase

    int* nznk;
    int* mznk;
    int* lznk;
};
#endif