#ifndef FFT_H
#define FFT_H

#include "input.h"


class FFT {
public:
    FFT();
    ~FFT();

    static bool fft_initialize(const int mm, const int n, FFT& fft);
    static void itoc(const int ik, const int mm, int *kk);
    static int  ctoi(const int *kk, const int mm); // * &???
    static void my_fft2d(const FFT& fft, const int n9, const double dx, const int kt, double **xr, double **xi);

    int     mm;
    int     n;
    int*    kk;
    int*    kj;
    int*    km0;
    int***  km;
    double* wr;
    double* wi;
};
#endif