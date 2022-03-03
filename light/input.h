#ifndef INPUT_H
#define INPUT_H
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <math.h>

using namespace std;

class Input
{
	public:
		Input();
		~Input();
		
		int mm;  
        int n_grid;
        int n9;
        int n1;

        int mgs;

        double a0;
        double xx0;
        double aa0;
        double dxy0;

        double plm;
        double zfh;

        double airy;
        double xxz;
        double aaz;
        double dxyz;

        int minZnkDim;
        int maxZnkOrder;

        double rms;
        double eeznk;
    
		bool INIT();
 } ;
 
 extern Input INPUT;
#endif
