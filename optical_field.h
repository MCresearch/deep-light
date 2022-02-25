#ifndef OPTICAL_FIELD
#define OPTICAL_FIELD

#include "fun.h"
#include "Zernike.h"
#include "input.h"
#include "fft.h"


class OPT
{
	public:
		OPT();
		~OPT();
		
		
		static bool Init_Intensity(OPT &opt);
		static bool Init_Phase(OPT &opt, double a1);
		static void numercial_diffraction(OPT &opt);

		double** ur;
        double** ui;
        int maxZnkDim;

        double* aznk;
        double* eznk;
        double* pl;
        double** ph;

		int* nznk;
		int* mznk;
		int* lznk;
	 
};
 #endif