#ifndef FFT_H
#define FFT_H

#include "input.h"


class FFT
{
    public:
        FFT();
        ~FFT();
		
		static bool fft_initialize(int mm, int n, FFT &fft);
        static void itoc(int ik, int* &kk, int mm);
        static int ctoi(int* &kk, int mm);
        static void my_fft2d(FFT &fft, int n9, double** &xr, double** &xi, double dx, int kt);

		int mm;
        int n;
        int* kk;
        int* kj;
        int* km0;
        int*** km;
        double* wr;
        double* wi;
	 
};
 #endif