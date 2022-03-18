#include "fft.h"
#include <iostream>
//#include <complex.h>
#include <fftw3.h>
#include <typeinfo>

//实部与虚部
#define REAL 0
#define IMAG 1

using namespace std;

void my_fft_1(int N, double** &ur, double** &ui)
{
	/*
	*fftw_complex is a FFTW custom complex class 是FFTW自定义的复数类
	*引入<complex>则会使用STL的复数类
	*/
	fftw_complex *in, *out;
    fftw_plan p;
    int k = 0;
    double* ur2;
    double* ui2;
    ur2 = new double[INPUT.n_grid*INPUT.n_grid]();
    ui2 = new double[INPUT.n_grid*INPUT.n_grid]();
    cout<<"001"<<endl;
    for(int i = 0; i < INPUT.n_grid; i++)
    {
        for(int j=0; j < INPUT.n_grid; j++)
        {
           ur2[k] = ur[i][j];
           ui2[k] = ui[i][j];
           k++;
        }

    }

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);

	if (in == NULL || out == NULL)
    {
        printf("ERROR!");
    }
    else
    {
        for (int i=0; i<N*N; i++)
        {
            
            in[i][REAL] = ur2[i];
            in[i][IMAG] = ui2[i];
           
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
    
    k = 0;
    int j = 0;
    int i = 0;
    for (int h = 0; h<N*N; h++)
    {
        ur[i][j] = out[h][0];
        ui[i][j] = out[h][1];
        j++;
        k++;
        if(k % N == 0)
        {
            i++;
            j = 0;
        }
    }
    if (in!=NULL)
        fftw_free(in);
    if (out!=NULL)
        fftw_free(out);

}


void my_fft_2(int N, double** &ur, double** &ui)
{
	/*
	*fftw_complex is a FFTW custom complex class 是FFTW自定义的复数类
	*引入<complex>则会使用STL的复数类
	*/
	fftw_complex *in, *out;
    fftw_plan p;
    int k = 0;
    double* ur2;
    double* ui2;
    ur2 = new double[INPUT.n_grid*INPUT.n_grid]();
    ui2 = new double[INPUT.n_grid*INPUT.n_grid]();

    for(int i = 0; i < INPUT.n_grid; i++)
    {
        for(int j=0; j < INPUT.n_grid; j++)
        {
           ur2[k] = ur[i][j];
           ui2[k] = ui[i][j];
           k++;
        }

    }

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);

	if (in == NULL || out == NULL)
    {
        printf("ERROR!");
    }
    else
    {
        for (int i=0; i<N*N; i++)
        {
            in[i][REAL] = ur2[i];
            in[i][IMAG] = ui2[i];
        }
    }

    p = fftw_plan_dft_2d(N, N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
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
    
    k = 0;
    int j = 0;
    int i = 0;
    for (int h = 0; h<N*N; h++)
    {
        ur[i][j] = out[h][0];
        ui[i][j] = out[h][1];
        j++;
        k++;
        if(k % N == 0)
        {
            i++;
            j = 0;
        }
    }
    if (in!=NULL)
        fftw_free(in);
    if (out!=NULL)
        fftw_free(out);

}
