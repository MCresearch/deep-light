#include "input.h"

void prop1(const int    n_grid,
           const int    n1,
           const double dz,
           const double kp,
           const double aa,
           double*      hr,
           double*      hi);
/** Transfer factor calculation

        The transmission factor from the initial square to the focal plane is calculated.

        @param n_grid       Number of grid-point
        @param n1           Number of grid-point/ 2 + 1
        @param dz           zzzz
        @param kp           wave_number
        @param aa           The spot radius of initial fields * Multiple of initial light field
        @param hr           the real part of transmission factor
        @param hi           the imaginary part of transmission factor
        @return             hr, hi
*/
void evol1(const int n_grid, const double* hr, const double* hi, double** ur, double** ui);
/** transmission

       The transmission factor is applied to the light field to complete the transmission.

        @param n_grid       Number of grid-point
        @param hr           the real part of transmission factor
        @param hi           the imaginary part of transmission factor
        @param ur           the real part of optical field
        @param ui           the imaginary part of optical field
        @return             ur, ui
*/

void mdfph(const int    n_grid,
           const int    n1,
           const double dx,
           const double dta,
           const double ddx,
           const double kp,
           double**     ur,
           double**     ui);
/** Coordinate transformation

        Resize the compute area grid.

        @param n_grid       Number of grid-point
        @param n1           Number of grid-point/ 2 + 1
        @param dx           Initial calculation area size
        @param dta          delta
        @param ddx          D
        @param kp           wave_number
        @param ur           the real part of optical field
        @param ui           the imaginary part of optical field
        @return             ur, ui
*/

void focusing(const int    n_grid,
              const int    n1,
              const double kp,
              const double dx,
              const double rzf,
              double**     ur,
              double**     ui);
/** foucusing

        Phase focusing

        @param n_grid       Number of grid-point
        @param n1           Number of grid-point/ 2 + 1
        @param kp           wave_number
        @param dx           Initial calculation area size
        @param rzf          1 / INPUT.zfh
        @param ur           the real part of optical field
        @param ui           the imaginary part of optical field
        @return             ur, ui
*/

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
void output_ui(const int n_grid, const string path, const int accuracy, double** ui);