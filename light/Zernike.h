#ifndef ZERNIKE
#define ZERNIKE

#include "input.h"

void rdm_gauss(double &a1, double &rdmg);
int  maxZernike(const int nk);
void nmlznk(const int maxZnkOrder, int &maxZnkDim, int *&nznk, int *&mznk, int *&lznk);
int  lznk_a(const int l, const int ni);
void zernike_cg(const int N, const double x, const double y, double *&zer);

#endif