#include "input.h"
#include "fun.h"
#include "Zernike.h"
#include "input.h"
#include "optical_field.h"

int main()
{
	cout << "111"<<endl;
	if(!INPUT.INIT())
	{
		cout << "input error!" << endl;
		exit(0); 
	}
	/*
	cout << INPUT.mm << endl;
	cout << INPUT.n_grid << endl;
	cout << INPUT.n9 << endl;
	cout << INPUT.n1 << endl;
	cout << INPUT.mgs << endl;
	cout << INPUT.a0 << endl;
	cout << INPUT.xx0 << endl;
	cout << INPUT.aa0 << endl;
	cout << INPUT.dxy0 << endl;
	
	cout << INPUT.plm << endl;
	cout << INPUT.zfh << endl;
	cout << INPUT.airy << endl;
	cout << INPUT.xxz << endl;
	cout << INPUT.aaz << endl;
	cout << INPUT.dxyz << endl;
	
	cout << INPUT.minZnkDim << endl;
	cout << INPUT.maxZnkOrder << endl;
	
	cout << INPUT.rms << endl;
	cout << INPUT.eeznk << endl;
	*/
	OPT opt;
	OPT::Init_Intensity(opt);

	double a1 = 0.0;
	a1 = 0.2391;		//随机数种子
	OPT::Init_Phase(opt, a1); // 此处在之后可加入循环

}

