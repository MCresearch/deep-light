#ifndef FFT_H
#define FFT_H

#include "input.h"


class FFT {
public:
    FFT();
    ~FFT();

    static bool fft_initialize(const int mm, const int n, FFT& fft);
    /** initialize the iterative indeies

        initialize km, km0, wr, wi
        km
        km0
        wr, wi          W^{k}_{N} = exp((i*2*PI/N)*k) k = -N ... N-1

        @param mm       mm = log_{2}(Number of grid-point)
        @param n        Number of grid-point
        @param fft
        @return         km, km0, wr, wi
    */

    static void itoc(const int ik, const int mm, int* kk);
    /** the binary code of an integer

        convert integers to binary

        @param ik       integer to be converted
        @param mm       mm = log_{2}(Number of grid-point)
        @param kk       converted binary array
        @return         kk
    */
    static int ctoi(const int* kk, const int mm);  // * &???
    /** the integer of a binary code

        convert binary to integers

        @param kk       binary array to be converted
        @param mm       mm = log_{2}(Number of grid-point)
        @return         converted integer
    */

    static void
    my_fft2d(const FFT& fft, const int n9, const double dx, const int kt, double** xr, double** xi);
    /** Fourier transform

        @param fft
        @param n9       Number of grid-point
        @param dx       the grid scale
        @param kt       kt > 0 : fourier transform;kt <= 0 : the inverse Fourier transform
        @param xr       The real part of the transformation
        @param xi       The imaginary part of the transformation
        @return         xr, xi
    */

    int     mm;  ///< mm = log_{2}(Number of grid-point)
    int     n;   ///< Number of grid-point
    int*    kk;
    int*    kj;
    int*    km0;
    int***  km;
    double* wr;
    double* wi;
};

#endif