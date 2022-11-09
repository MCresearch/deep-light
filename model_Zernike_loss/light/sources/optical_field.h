#ifndef OPTICAL_FIELD
#define OPTICAL_FIELD

#include "FFt.h"
#include "Zernike.h"
#include "fun.h"
#include "input.h"

class OPT {
public:
  OPT();
  ~OPT();

  static bool Init_Intensity(Input &INPUT, OPT &opt);
  static bool Init_Phase(Input &INPUT, OPT &opt, double a1, double **a, int num,
                         const string type,string dir0);
  static void numercial_diffraction(Input &INPUT, const double a1, OPT &opt,string dir0);

  double **ur0; ///< The real part of the transformation
  double **ui0; ///< The imaginary part of the transformation

  double **ur;   ///< The real part of the transformation
  double **ui;   ///< The imaginary part of the transformation
  int maxZnkDim; ///< Maximum order of a polynomial.

  double *aznk;
  double *eznk;
  double *pl;  ///< cg Zernike coefficient
  double **ph; ///< phase

  int *nznk;
  int *mznk;
  int *lznk;
};
#endif