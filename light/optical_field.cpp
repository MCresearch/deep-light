#include "optical_field.h"
#include "FFt.h"
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


	// set inIntensity
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
	// save inIntensity
	ofstream outfile1;
	outfile1.open("./test/dl_inIntensity.dat", ios::app);
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
//Init_Phase a1 is seed  
// Polynomial coefficient: Gaussian random number 
	//a1 = 0.2391;		
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
	// Phase
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

	
	for(int i = 0; i < opt.maxZnkDim; i++)
	{
		opt.pl[i] = 0;
	}

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
		//cout << i << "\t"<<rdmg[i] << endl;
		opt.eznk[i] = exp(-opt.nznk[i] * INPUT.eeznk);
		opt.aznk[i] = rdmg[i] * opt.eznk[i];
		ss = ss + pow(opt.aznk[i],2);
	}
		
	//系数按方差rms归一化
	for(int i=3; i <= opt.maxZnkDim; i++)
	{
		opt.aznk[i] = opt.aznk[i] * sqrt(INPUT.rms/ss);
	}
	ofstream outfile2;
	outfile2.open("./test/dl_zernike_coeff.dat", ios::app);
	outfile2.setf(ios::fixed, ios::floatfield); 
    outfile2.precision(7);  
	if(!outfile2.is_open())
	{
		cout << "./test/open file failure" << endl;
	}
	for(int i = 1; i <= opt.maxZnkDim; i++)
	{
		outfile2 << i << "\t" << opt.aznk[i]  << '\t' << opt.nznk[i] << "\t" << opt.eznk[i]<< endl;
	}
	outfile2.close();

	/*
	ofstream outfile23;
	outfile23.open("./test/dl_pl_aznk.dat", ios::app);

	//设置相位
	for(int j = 0; j < INPUT.n_grid; j++)
	{
		y = (j+1-INPUT.n1)*INPUT.dxy0;
		y2 = y * y;
		for(int i = 0; i < INPUT.n_grid; i++)
		{
			x = (i+1-INPUT.n1)*INPUT.dxy0;			
			x2 = x * x;
			r2 = x2 + y2;
			if(r2/a02 <= 1)
			{
				zernike_cg(opt.maxZnkDim, opt.pl, x/INPUT.a0, y/INPUT.a0);

				ofstream outfile22;
				outfile22.open("./test/dl_zernike_cg.dat", ios::app);
				//outfile22.open("./test/dl_x_y.dat", ios::app);
				outfile22.setf(ios::fixed, ios::floatfield); 
    			outfile22.precision(7);  
				//outfile22 << i << "\t" << j << "\t" <<  x/INPUT.a0 << "\t" <<  y/INPUT.a0<< endl;
				
				if(!outfile22.is_open())
				{
					cout << "./test/open file failure" << endl;
				}
				for(int i = 1; i <= opt.maxZnkDim; i++)
				{
					outfile22 << i << "\t" << opt.pl[i]  << endl;
				}
				outfile22.close();

				opt.ph[i][j] = 0;
				for(int l = INPUT.minZnkDim; l <= opt.maxZnkDim; l++)
				{
					opt.ph[i][j] = opt.ph[i][j] + opt.pl[l] * opt.aznk[l];
					outfile23 << l << "\t" << opt.pl[l]  << "\t" << opt.aznk[l]  << endl;
				}
				uri = opt.ur[i][j];
				opt.ur[i][j] = uri*cos(opt.ph[i][j]);
				opt.ui[i][j]= uri*sin(opt.ph[i][j]);
			}		
		}
	} 

	outfile23.close();
	*/
	//test********************************************************8
	ifstream ifs1("./test/inPhase.dat");
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
				ifs1 >> opt.ph[i][j];
				uri = opt.ur[i][j];
				opt.ur[i][j] = uri*cos(opt.ph[i][j]);
				opt.ui[i][j]= uri*sin(opt.ph[i][j]);	
		}
	} 
	ifs1.close();

//test*******************************************************
	ofstream outfile3;
	ofstream outfile31;
	outfile3.open("./test/dl_inPhase.dat", ios::app);
	outfile31.open("./test/dl_inPhase_intensity.dat", ios::app);
	outfile31.setf(ios::fixed, ios::floatfield); 
    outfile31.precision(6);  
	outfile3.setf(ios::fixed, ios::floatfield); 
    outfile3.precision(6);  
	if(!outfile3.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile3 << opt.ph[i][j] << '\t';
			outfile31 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile3 <<  endl;
		outfile31 <<  endl;
	}
	cout << "ph"<<opt.ph[77][129] << endl;
	outfile3.close();
	outfile31.close();

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
	FFT fft;
	hr = new double[INPUT.n_grid]();
	hi = new double[INPUT.n_grid]();

	FFT::fft_initialize(INPUT.mm, INPUT.n_grid, fft);

	ofstream outfile4;
	outfile4.open("./test/dl_fft_initialize.dat", ios::app);
	outfile4.setf(ios::fixed, ios::floatfield); 
    outfile4.precision(6);  
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

	dxy0 = INPUT.aa0 / INPUT.n_grid;
	dxyz = INPUT.aaz / INPUT.n_grid;

	dlta = (1 - INPUT.aaz / INPUT.aa0) /INPUT.zfh;
	ddxz = 1 - dlta * INPUT.zfh;
  
	dk0 = 1 / INPUT.aa0;
	zzzz = INPUT.zfh / (1 - dlta * INPUT.zfh);
	PI = 3.141592653589793;
	wave_number = 2 * PI / INPUT.plm;	

	focusing(INPUT.n_grid, INPUT.n1, opt.ur, opt.ui, wave_number, dxy0, 1 / INPUT.zfh);

	ofstream outfile5;
	outfile5.open("./test/dl_focusing.dat", ios::app);
	outfile5.setf(ios::fixed, ios::floatfield); 
    outfile5.precision(6);  
	if(!outfile5.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile5 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile5 <<  endl;
	}
	outfile5.close();

	mdfph(INPUT.n_grid, INPUT.n1, opt.ur, opt.ui, dxy0, dlta, 1, wave_number);
	
	ofstream outfile6;
	outfile6.open("./test/dl_mdfph1.dat", ios::app);
	outfile6.setf(ios::fixed, ios::floatfield); 
    outfile6.precision(6);  
	if(!outfile6.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile6 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile6 <<  endl;
	}
	outfile6.close();
	

	FFT::my_fft2d(fft, INPUT.n_grid, opt.ur, opt.ui, dxy0, 2);

	ofstream outfile7;
	outfile7.open("./test/dl_my_fft2d1.dat", ios::app);
	outfile7.setf(ios::fixed, ios::floatfield); 
    outfile7.precision(6);  
	if(!outfile7.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile7 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile7 <<  endl;
	}
	outfile7.close();
	
	
	prop1(INPUT.n_grid, INPUT.n1, hr, hi, zzzz, wave_number, INPUT.aa0);

	ofstream outfile8;
	outfile8.open("./test/dl_prop1.dat", ios::app);
	outfile8.setf(ios::fixed, ios::floatfield); 
    outfile8.precision(6);  
	if(!outfile8.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile8 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile8 <<  endl;
	}
	outfile8.close();

	evol1(INPUT.n_grid, hr, hi, opt.ur, opt.ui);

	ofstream outfile9;
	outfile9.open("./test/dl_evol1.dat", ios::app);
	outfile9.setf(ios::fixed, ios::floatfield); 
    outfile9.precision(6);  
	if(!outfile9.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile9 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile9 <<  endl;
	}
	outfile9.close();

	FFT::my_fft2d(fft, INPUT.n_grid, opt.ur, opt.ui, dk0, -2);
	cout << "dxy0" << dxy0<< endl;
	cout << "dk0" << dk0 << endl;
	ofstream outfile10;
	outfile10.open("./test/dl_my_fft2d2.dat", ios::app);
	outfile10.setf(ios::fixed, ios::floatfield); 
    outfile10.precision(6);  
	if(!outfile10.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile10 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile10 <<  endl;
	}
	outfile10.close();

	mdfph(INPUT.n_grid, INPUT.n1, opt.ur, opt.ui, dxyz, dlta, ddxz, - wave_number);

	ofstream outfile11;
	outfile11.open("./test/dl_mdfph2.dat", ios::app);
	//outfile11.setf(ios::fixed, ios::floatfield); 
    //outfile11.precision(6);  
	if(!outfile11.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile11 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile11 <<  endl;
	}
	outfile11.close();

	pkkz = 1.0/ ddxz / ddxz;
	for(int j = 0; j < INPUT.n_grid; j++)
	{
		for(int i = 0; i < INPUT.n_grid; i++)
		{
			opt.ur[i][j] = opt.ur[i][j]/ddxz;
	    	opt.ui[i][j] = opt.ui[i][j]/ddxz;
		}
	}

	ofstream outfile12;
	outfile12.open("./test/dl_outIntensity.dat", ios::app);
	//outfile12.setf(ios::fixed, ios::floatfield); 
    //outfile12.precision(6);  
	if(!outfile12.is_open())
	{
		cout << "open file failure" << endl;
	}
	for(int i = 0; i < INPUT.n_grid; i++)
	{
		for(int j = 0; j < INPUT.n_grid; j++)
		{
			outfile12 << pow(opt.ur[i][j],2)+pow(opt.ui[i][j],2) << '\t';
		}
		outfile12 <<  endl;
	}
	outfile12.close();
	
}
	