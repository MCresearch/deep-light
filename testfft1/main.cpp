#include <iostream>

#include "fftw3.h"


//实部与虚部
#define REAL 0
#define IMAG 1

using namespace std;

int main() {
	
    fftw_plan p;
	int N = 5;
	fftw_complex **in, **out;
    in = (fftw_complex**) fftw_malloc( N*sizeof(fftw_complex *));
    out = (fftw_complex**) fftw_malloc(N*sizeof(fftw_complex *));
	for (int i = 0; i < N; ++i) 
	{
     	in[i] = (fftw_complex *) malloc(N * sizeof(fftw_complex));
	}
	double a[5][5];
	a[0][0] = 11;

	in[1][1][0]= a[0][0];
	cout << sizeof(fftw_complex)<< endl;
	cout << in[1][1][0]<< endl;
}
/*
fftw_complex ***a_bad_array;  //another way to make a 5x12x27 array 
a_bad_array = (fftw_complex ***) malloc(5 * sizeof(fftw_complex **));
for (i = 0; i < 5; ++i) {
     a_bad_array[i] = 
        (fftw_complex **) malloc(12 * sizeof(fftw_complex *));
     for (j = 0; j < 12; ++j)
          a_bad_array[i][j] =
                (fftw_complex *) malloc(27 * sizeof(fftw_complex));
}
*/

