#include "input.h"

void prop1(int n_grid,int n1, double* &hr, double* &hi, double dz, double kp, double aa);
void evol1(int n_grid,double* &hr, double* &hi, double** &ur, double** &ui);
void mdfph(int n_grid,int n1, double** &ur, double** &ui, double dx, double dta, double ddx, double kp);
void focusing(int n_grid,int n1, double** &ur, double** &ui, double kp, double dx, double rzf);