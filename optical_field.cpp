#include "optical_field.h"
#include <random>
#include <fstream>
#include <sstream>

OPT::OPT(){}

OPT::~OPT(){}

bool OPT::Init_Intensity(OPT &opt)
{
	double a02 = 0.0;
	double x = 0.0;
	double y = 0.0;
	double x2 = 0.0;
	double y2 = 0.0;
	double r2 = 0.0;
	double uri = 0.0;
	double a1 = 0.0;
	double rdmg = 0.0;
	double ss = 0.0;

	opt.ur = new double*[INPUT.n_grid]();
	opt.ui = new double*[INPUT.n_grid]();

	for(int i = 0; i < INPUT.n_grid; i++)
	{
		opt.ur[i] = new double[INPUT.n_grid]();
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		opt.ui[i] = new double[INPUT.n_grid]();
	}


	// 设置光强
	a02 = INPUT.a0*INPUT.a0;
	for(int j = 0; j < INPUT.n_grid; j++)
	{
		y = (j+1-INPUT.n1)*INPUT.dxy0;  // +不加1
		y2 = y * y;
		for(int i = 0; i < INPUT.n_grid; i++)
		{
			x = (i+1-INPUT.n1)*INPUT.dxy0;
			x2 = x*x;
			r2 = x2+y2;
			opt.ur[i][j] = exp(-1 * pow(r2/a02, INPUT.mgs));
			opt.ui[i][j] = 0;
		}
	}
	// 存储光强
	ofstream outfile1;
	outfile1.open("inIntensity.dat", ios::app);
	outfile1.setf(ios::fixed, ios::floatfield); 
    outfile1.precision(6);  
	if(!outfile1.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile1 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile1 <<  endl;
	}
	outfile1.close();
}


bool OPT::Init_Phase(OPT &opt, double a1)
{
//初始化相位，a1为随机数种子，a02为
// 多项式系数：高斯随机数
	//a1 = 0.2391;		//随机数初值
    double x = 0.0;
	double y = 0.0;
	double x2 = 0.0;
	double y2 = 0.0;
	double r2 = 0.0;
	double uri = 0.0;
    double ss = 0.0;
    //double rdmg = 0.0;
	double a02 = 0.0;
	ss = 0;
	a02 = INPUT.a0*INPUT.a0;
	// 相位
	opt.maxZnkDim = maxZernike(INPUT.maxZnkOrder);
	cout << "maxZnkDim =" << opt.maxZnkDim << endl;
	opt.nznk = new int[opt.maxZnkDim]();
	opt.mznk = new int[opt.maxZnkDim]();
	opt.lznk = new int[opt.maxZnkDim]();
	nmlznk(INPUT.maxZnkOrder,opt.maxZnkDim,opt.nznk,opt.mznk,opt.lznk);

	opt.aznk = new double[opt.maxZnkDim]();
	opt.eznk = new double[opt.maxZnkDim]();
	opt.ph = new double*[INPUT.n_grid]();
	opt.pl = new double[opt.maxZnkDim]();

	for(int i = 0; i < INPUT.n_grid; i++)
	{
		opt.ph[i] = new double[INPUT.n_grid]();
	}

	default_random_engine random(a1);
    std::normal_distribution<double> dis(0,1);  

	double rdmg[200];
	ifstream ifs("rdmg.dat");
	for(int i=3; i <= opt.maxZnkDim; i++)
	{
		ifs >> rdmg[i];
	}
	ifs.close();

	for(int i=3; i <= opt.maxZnkDim; i++)
	{
		//rdmg = dis(random);
		//rdm_gauss(a1,rdmg);
		opt.eznk[i] = exp(-opt.nznk[i] * INPUT.eeznk);
		opt.aznk[i] = rdmg[i] * opt.eznk[i];
		ss = ss + pow(opt.aznk[i],2);
	}
		
	//系数按方差rms归一化
	for(int i=3; i <= opt.maxZnkDim; i++)
	{
		opt.aznk[i] = opt.aznk[i] * sqrt(INPUT.rms/ss);
	}
	/*
	open(11,file='zernike_coeff.dat')
	do i=1,maxZnkDim
	write(11,*)i,aznk(i),nznk(i),eznk(i)
	end do
	close(1)
	*/
	ofstream outfile2;
	outfile2.open("zernike_coeff.dat", ios::app);
	//outfile2.setf(ios::fixed, ios::floatfield); 
    //outfile2.precision(6);  
	if(!outfile2.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 1; i <= opt.maxZnkDim; i++)
	{
		outfile2 << i << "\t" << opt.aznk[i]  << '\t' << opt.nznk[i] << "\t" << opt.eznk[i]<< endl;
	}
	outfile2.close();

	//设置相位
	for(int j = 0; j < INPUT.n_grid; j++)
	{
		y = (j-INPUT.n1)*INPUT.dxy0;
		y2 = y * y;
		for(int i = 0; i < INPUT.n_grid; i++)
		{
			x = (i-INPUT.n1)*INPUT.dxy0;			
			x2 = x * x;
			r2 = x2 + y2;
			if(r2/a02 <= 1)
			{
				zernike_cg(opt.maxZnkDim, opt.pl, x/INPUT.a0, y/INPUT.a0);
				opt.ph[i][j] = 0;
				for(int l = INPUT.minZnkDim; l <= opt.maxZnkDim; l++)
				{
					opt.ph[i][j] = opt.ph[i][j] + opt.pl[l] * opt.aznk[l];
				}
				uri = opt.ur[i][j];
				opt.ur[i][j] = uri*cos(opt.ph[i][j]);
				opt.ui[i][j]= uri*sin(opt.ph[i][j]);
			}		
		}
	} 
	ofstream outfile3;
	outfile3.open("inPhase.dat", ios::app);
	//outfile3.setf(ios::fixed, ios::floatfield); 
    //outfile3.precision(6);  
	if(!outfile3.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile3 << opt.ph[i][j] << '\t';
		}
		outfile3 <<  endl;
	}
	outfile3.close();


	delete []opt.aznk;
	delete []opt.eznk;  
	delete []opt.pl;  
	for(int i = 0; i < opt.maxZnkDim; i++)
		delete[] opt.ph[i];
	delete[] opt.ph;
}


void OPT::numercial_diffraction(OPT &opt)
{
	double dxy0 = 0.0;
	double dxyz = 0.0;
	double dlta = 0.0;
	double ddxz = 0.0;
	double dk0 = 0.0;
	double zzzz = 0.0;
	double wave_number = 0.0;
	double pkkz = 0.0;
	double PI = 3.141592653589793;

	double* hr;
	double* hi;
	hr = new double[INPUT.n_grid]();
	hi = new double[INPUT.n_grid]();

	for(int i = 0; i < INPUT.n_grid; i++)
	{
		opt.ur[i] = new double[INPUT.n_grid]();
	}

	dxy0 = INPUT.aa0 / INPUT.n_grid;
	dxyz = INPUT.aaz / INPUT.n_grid;

	dlta = (1 - INPUT.aaz / INPUT.aa0) /INPUT.zfh;
	ddxz = 1 - dlta * INPUT.zfh;
  
	dk0 = 1 / INPUT.aa0;
	zzzz = INPUT.zfh / (1 - dlta * INPUT.zfh);
	PI = atan(1)*4;
	wave_number = 2 * PI / INPUT.plm;	

	

	focusing(INPUT.n_grid, INPUT.n9, INPUT.n1, opt.ur, opt.ui, wave_number, dxy0, 1 / INPUT.zfh);

	mdfph(INPUT.n_grid, INPUT.n9, INPUT.n1, opt.ur, opt.ui, dxy0, dlta, 1, wave_number);
	
	my_fft_1(INPUT.n9, opt.ur, opt.ui);

	ofstream outfile4;
	outfile4.open("urfft.dat", ios::app);
	//outfile4.setf(ios::fixed, ios::floatfield); 
    //outfile4.precision(6);  
	if(!outfile4.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile4 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile4 <<  endl;
	}
	outfile4.close();


	prop1(INPUT.n_grid, INPUT.n9, INPUT.n1, hr, hi, zzzz, wave_number, INPUT.aa0);

	evol1(INPUT.n_grid, INPUT.n9, hr, hi, opt.ur, opt.ui);

	my_fft_2(INPUT.n9, opt.ur, opt.ui);

	mdfph(INPUT.n_grid, INPUT.n9, INPUT.n1, opt.ur, opt.ui, dxyz, dlta, ddxz, - wave_number);

	pkkz = 1.0/ ddxz / ddxz;
	for(int j = 0; j < INPUT.n_grid; j++)
	{
		for(int i = 0; i < INPUT.n_grid; i++)
		{
			opt.ur[i][j] = opt.ur[i][j]/ddxz;
	    	opt.ui[i][j] = opt.ui[i][j]/ddxz;
		}
	}
}
	