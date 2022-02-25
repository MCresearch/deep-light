//==================================================
// Main function：The far-field transmission of focused beam is realized by fast Fourier transform and coordinate adaptation transform.
// Date: 2022-02-14
//==================================================
#include "input.h"
#include "fun.h"
#include "Zernike.h"
#include "input.h"
#include "optical_field.h"
#include<time.h>


int main()
{
	cout << "111"<<endl;
	if(!INPUT.INIT())
	{
		cout << "input error!" << endl;
		exit(0); 
	}

	OPT opt;
	OPT::Init_Intensity(opt);

	double a1 = 0.0;
	a1 = 0.2391;		//随机数种子
	OPT::Init_Phase(opt, a1); // 此处在之后可加入循环
	OPT::numercial_diffraction(opt);

	// 输出出结果时刻
	cout << "The current time is: " <<(double)clock()<< "s" << endl;
	// 输出运行时间; 
	cout << "The run time is: " <<(double)clock() / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}

