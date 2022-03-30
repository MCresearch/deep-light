#include "input.h"

void prop1(const int    n_grid,
           const int    n1,
           const double dz,
           const double kp,
           const double aa,
           double*      hr,
           double*      hi);
void evol1(const int n_grid, const double* hr, const double* hi, double** ur, double** ui);
void mdfph(const int    n_grid,
           const int    n1,
           const double dx,
           const double dta,
           const double ddx,
           const double kp,
           double**     ur,
           double**     ui);
void focusing(const int    n_grid,
              const int    n1,
              const double kp,
              const double dx,
              const double rzf,
              double**     ur,
              double**     ui);
void output_inIntensity(const int    n_grid,
                        const string path,
                        const int    accuracy,
                        double**     ur,
                        double**     ui);
void output_zernike_coeff(const int     n_grid,
                          const string  path,
                          const int     accuracy,
                          const int     maxZnkDim,
                          const double* aznk,
                          const int*    nznk,
                          const double* eznk);
void output_inPhase(const int n_grid, const string path, const int accuracy, double** ph);
void output_ur(const int n_grid, const string path, const int accuracy, double** ur);