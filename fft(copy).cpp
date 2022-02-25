#include "fft.h"
#include <iostream>
//#include <complex.h>
#include <fftw3.h>
#include <typeinfo>
/*
//实部与虚部
#define REAL 0
#define IMAG 1
*/
using namespace std;

void my_fft_1(int N, double** &ur, double** &ui)
{
	/*
	*fftw_complex 是FFTW自定义的复数类
	*引入<complex>则会使用STL的复数类
	*/
    fftw_complex **in, **out;
    fftw_plan p;

    in = (fftw_complex**) fftw_malloc( N*sizeof(fftw_complex *));
    out = (fftw_complex**) fftw_malloc(N*sizeof(fftw_complex *));
	for (int i = 0; i < N; ++i) 
	{
     	in[i] = (fftw_complex *) malloc(N * sizeof(fftw_complex));
         out[i] = (fftw_complex *) malloc(N * sizeof(fftw_complex));
	}
    
    if (in == NULL || out == NULL)
    {
        printf("ERROR!");
    }
    else
    {
        for (int i=0; i<N; i++)
        {
            for (int j=0; j<N; j++)
            {
                in[i][j][0] = ur[i][j];
                in[i][j][1] = ui[i][j];
            }
            
        }
    }

    p = fftw_plan_dft_2d(N, N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup();

    for (int i=0; i<N*N; i++)
    {
        printf("%f, %fi\n", in[i][0], in[i][1]);
    }
    cout << endl;
    for (int i=0; i<N*N; i++)
    {
        printf("%f, %fi\n", out[i][0], out[i][1]);
    }
    
    if (in!=NULL)
        fftw_free(in);
    if (out!=NULL)
        fftw_free(out);
}


void my_fft_2(int N, double** &ur, double** &ui)
{
	/*
	*fftw_complex 是FFTW自定义的复数类
	*引入<complex>则会使用STL的复数类
	*/
     fftw_complex **in, **out;
    fftw_plan p;

    in = (fftw_complex**) fftw_malloc( N*sizeof(fftw_complex *));
    out = (fftw_complex**) fftw_malloc(N*sizeof(fftw_complex *));
	for (int i = 0; i < N; ++i) 
	{
     	in[i] = (fftw_complex *) malloc(N * sizeof(fftw_complex));
	}
    for (int i = 0; i < N; ++i) 
	{
     	out[i] = (fftw_complex *) malloc(N * sizeof(fftw_complex));
	}

	if(in == NULL || out == NULL)
    {
        printf("ERROR!");
    }
    else
    {
        for(int i=0; i<N*N; i++)
        {
            for (int j=0; j<N*N; j++)
            {
                in[i][j][0] = ur[i][j];
                in[i][j][1] = ui[i][j];
            }
        }
    }

    p = fftw_plan_dft_2d(N, N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup();

    for(int i=0; i<N*N; i++)
    {
        printf("%f, %fi\n", in[i][0], in[i][1]);
    }
    cout << endl;
    for (int i=0; i<N*N; i++)
    {
        printf("%f, %fi\n", out[i][0], out[i][1]);
    }
    if (in!=NULL)
        fftw_free(in);
    if (out!=NULL)
        fftw_free(out);
}

