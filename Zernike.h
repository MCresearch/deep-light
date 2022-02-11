#ifndef ZERNIKE
#define ZERNIKE

#include "input.h"

int maxZernike(int nk);
void nmlznk(int maxZnkOrder,int &maxZnkDim, int* &nznk,int* &mznk,int* &lznk);
int lznk_a(int l, int ni);
void zernike_cg(int N, double* &zer, double x, double y);

 #endif