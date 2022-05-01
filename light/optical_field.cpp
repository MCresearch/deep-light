#include "optical_field.h"
#include "FFt.h"
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#define PI 3.141592653589793

OPT::OPT() {}

OPT::~OPT() {}

bool OPT::Init_Intensity(Input &INPUT, OPT &opt)
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

    opt.ur = new double *[INPUT.n_grid]();
    opt.ui = new double *[INPUT.n_grid]();

    for (int i = 0; i < INPUT.n_grid; i++)
    {
        opt.ur[i] = new double[INPUT.n_grid]();
    }
    for (int i = 0; i < INPUT.n_grid; i++)
    {
        opt.ui[i] = new double[INPUT.n_grid]();
    }


    // set inIntensity
    a02 = INPUT.a0 * INPUT.a0;
    for (int i = 0; i < INPUT.n_grid; i++)
    {
        x = (i + 1 - INPUT.n1) * INPUT.dxy0;  // +不加1
        x2 = x * x;
        for (int j = 0; j < INPUT.n_grid; j++)
        {
            y = (j + 1 - INPUT.n1) * INPUT.dxy0;
            y2 = y * y;
            r2 = x2 + y2;
            opt.ur[i][j] = exp(-1 * pow(r2 / a02, INPUT.mgs));
            opt.ui[i][j] = 0;
        }
    }
    // save inIntensity
    if (INPUT.out_inIntensity == 1)
    {
        output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_inIntensity.dat", 6, opt.ur, opt.ui);
        output_ur(INPUT.n_grid, INPUT.dir + "dl_inIntensity.dat", 6, opt.ur);
        output_ui(INPUT.n_grid, INPUT.dir + "dl_inIntensity.dat", 6, opt.ui);
    }
}


bool OPT::Init_Phase(Input &INPUT, OPT &opt, double a1, const string type)
{
    // Init_Phase a1 is seed
    // Polynomial coefficient: Gaussian random number
    // a1 = 0.2391;
    double x = 0.0;
    double y = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    double r2 = 0.0;
    double uri = 0.0;
    double ss = 0.0;
    // double rdmg = 0.0;
    double a02 = 0.0;

    char buffer[20];
    sprintf(buffer, "%f", a1);
    string str = buffer;

    a02 = INPUT.a0 * INPUT.a0;
    // Phase
    opt.maxZnkDim = maxZernike(INPUT.maxZnkOrder);
    cout << "maxZnkDim =" << opt.maxZnkDim << endl;
    opt.nznk = new int[opt.maxZnkDim]();
    opt.mznk = new int[opt.maxZnkDim]();
    opt.lznk = new int[opt.maxZnkDim]();
    nmlznk(INPUT.maxZnkOrder, opt.maxZnkDim, opt.nznk, opt.mznk, opt.lznk);  // delete lznk?

    opt.aznk = new double[opt.maxZnkDim]();
    opt.eznk = new double[opt.maxZnkDim]();
    opt.ph = new double *[INPUT.n_grid]();
    opt.pl = new double[opt.maxZnkDim]();


    for (int i = 0; i < opt.maxZnkDim; i++)
    {
        opt.pl[i] = 0;
    }

    for (int i = 0; i < INPUT.n_grid; i++)
    {
        opt.ph[i] = new double[INPUT.n_grid]();
    }
    if (type == "random")
    {
        default_random_engine            random(a1);
        std::normal_distribution<double> dis(0, 1);
        /*
        double   rdmg[200];
        ifstream ifs("rdmg.dat");
        for (int i = 3; i <= opt.maxZnkDim; i++)
        {
            ifs >> rdmg[i];
        }
        ifs.close();
        */
        double rdmg = 0;
        for (int i = 3; i <= opt.maxZnkDim; i++)
        {
            //rdmg = dis(random);
            rdm_gauss(a1, rdmg);
            // cout << i << "\t" << rdmg << endl;
            opt.eznk[i] = exp(-opt.nznk[i] * INPUT.eeznk);
            opt.aznk[i] = rdmg * opt.eznk[i];
            // opt.aznk[i] = rdmg[i] * opt.eznk[i];
            ss = ss + pow(opt.aznk[i], 2);
        }

        //系数按方差rms归一化
        for (int i = 3; i <= opt.maxZnkDim; i++)
        {
            opt.aznk[i] = opt.aznk[i] * sqrt(INPUT.rms / ss);
        }
        if (INPUT.out_zernike_coeff == 1)
        {
            output_zernike_coeff(INPUT.n_grid, INPUT.dir + "dl_zernike_coeff_" + str + ".dat", 7,
                                 opt.maxZnkDim, opt.aznk, opt.nznk, opt.eznk);
        }
        //ofstream outfile23;
        //outfile23.open("./tests/dl_pl_aznk.dat", ios::app);

        //设置相位
        for (int i = 0; i < INPUT.n_grid; i++)
        {
            x = (i + 1 - INPUT.n1) * INPUT.dxy0;
            x2 = x * x;
            for (int j = 0; j < INPUT.n_grid; j++)
            {
                y = (j + 1 - INPUT.n1) * INPUT.dxy0;
                y2 = y * y;
                r2 = x2 + y2;
                if (r2 / a02 <= 1)
                {
                    zernike_cg(opt.maxZnkDim, x / INPUT.a0, y / INPUT.a0, opt.pl);
                    opt.ph[i][j] = 0;
                    for (int l = INPUT.minZnkDim; l <= opt.maxZnkDim; l++)
                    {
                        opt.ph[i][j] = opt.ph[i][j] + opt.pl[l] * opt.aznk[l];
                        //outfile23 << l << "\t" << opt.pl[l] << "\t" << opt.aznk[l] << endl;
                    }
                    uri = opt.ur[i][j];
                    opt.ur[i][j] = uri * cos(opt.ph[i][j]);
                    opt.ui[i][j] = uri * sin(opt.ph[i][j]);
                }
            }
        }
        //outfile23.close();
    }

    else if (type == "confirm")
    {
        cout << "confirm" << endl;
        ifstream ifs1(INPUT.dir + "inPhase.dat");
        for (int j = 0; j < INPUT.n_grid; j++)
        {
            for (int i = 0; i < INPUT.n_grid; i++)
            {
                ifs1 >> opt.ph[i][j];
                uri = opt.ur[i][j];
                opt.ur[i][j] = uri * cos(opt.ph[i][j]);
                opt.ui[i][j] = uri * sin(opt.ph[i][j]);
            }
        }
        ifs1.close();
    }
    else
    {
        cout << "input phase type error!" << endl;
        exit(0);
    }

    if (INPUT.out_inPhase == 1)
    {
        output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_inPhase_" + str + ".dat", 6, opt.ur,
                           opt.ui);
        // output_inPhase(INPUT.n_grid, INPUT.dir + "dl_inPhase_" + str + ".dat", 6, opt.ph);
        output_ur(INPUT.n_grid, INPUT.dir + "dl_inPhase_" + str + ".dat", 6, opt.ur);
        output_ui(INPUT.n_grid, INPUT.dir + "dl_inPhase_" + str + ".dat", 6, opt.ui);
    }
    delete[] opt.aznk;
    delete[] opt.eznk;
    delete[] opt.pl;
    for (int i = 0; i < INPUT.n_grid; i++)
    {
        delete[] opt.ph[i];
    }
    delete[] opt.ph;
}


void OPT::numercial_diffraction(Input &INPUT, const double a1, OPT &opt)
{
    cout << "numercial_diffraction" << endl;
    double  dxy0 = 0.0;
    double  dxyz = 0.0;
    double  dlta = 0.0;
    double  ddxz = 0.0;
    double  dk0 = 0.0;
    double  zzzz = 0.0;
    double  wave_number = 0.0;
    double  pkkz = 0.0;
    double *hr;
    double *hi;
    FFT     fft;

    char buffer[20];
    sprintf(buffer, "%f", a1);
    string str = buffer;

    hr = new double[INPUT.n_grid]();
    hi = new double[INPUT.n_grid]();
    cout << "111" << endl;
    FFT::fft_initialize(INPUT.mm, INPUT.n_grid, fft);
    // cout << "222" << endl;
    // output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_fft_initialize.dat", 6, opt.ur, opt.ui);

    dxy0 = INPUT.aa0 / INPUT.n_grid;
    dxyz = INPUT.aaz / INPUT.n_grid;

    dlta = (1 - INPUT.aaz / INPUT.aa0) / INPUT.zfh;
    ddxz = 1 - dlta * INPUT.zfh;

    dk0 = 1 / INPUT.aa0;
    zzzz = INPUT.zfh / (1 - dlta * INPUT.zfh);
    wave_number = 2 * PI / INPUT.plm;

    // output_ur(INPUT.n_grid, INPUT.dir + "dl_ur_" + str + ".dat", 6, opt.ur);
    // output_ur(INPUT.n_grid, INPUT.dir + "dl_ui_" + str + ".dat", 6, opt.ui);

    focusing(INPUT.n_grid, INPUT.n1, wave_number, dxy0, 1 / INPUT.zfh, opt.ur, opt.ui);
    if (INPUT.out_focusing == 1)
    {
        output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_focusing_" + str + ".dat", 6, opt.ur,
                           opt.ui);
        output_ur(INPUT.n_grid, INPUT.dir + "dl_focusing_" + str + ".dat", 6, opt.ur);
        output_ui(INPUT.n_grid, INPUT.dir + "dl_focusing_" + str + ".dat", 6, opt.ui);
    }
    mdfph(INPUT.n_grid, INPUT.n1, dxy0, dlta, 1, wave_number, opt.ur, opt.ui);
    if (INPUT.out_mdfph1 == 1)
    {
        output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_mdfph1_" + str + ".dat", 6, opt.ur,
                           opt.ui);
        // cout << "dxy0" << dxy0 << "dlta" << dlta << "wave_number" << wave_number << endl;
        output_ur(INPUT.n_grid, INPUT.dir + "dl_mdfph1_" + str + ".dat", 6, opt.ur);
        output_ui(INPUT.n_grid, INPUT.dir + "dl_mdfph1_" + str + ".dat", 6, opt.ui);
    }
    FFT::my_fft2d(fft, INPUT.n_grid, dxy0, 2, opt.ur, opt.ui);
    if (INPUT.out_my_fft2d1 == 1)
    {
        output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_my_fft2d1_" + str + ".dat", 6, opt.ur,
                           opt.ui);
        output_ur(INPUT.n_grid, INPUT.dir + "dl_my_fft2d1_" + str + ".dat", 6, opt.ur);
        output_ui(INPUT.n_grid, INPUT.dir + "dl_my_fft2d1_" + str + ".dat", 6, opt.ui);
    }

    prop1(INPUT.n_grid, INPUT.n1, zzzz, wave_number, INPUT.aa0, hr, hi);
    /*
    ofstream outfile1;
    ofstream outfile2;
    outfile1.open(INPUT.dir + "dl_prop1_hr_" + str + ".dat", ios::app);
    outfile2.open(INPUT.dir + "dl_prop1_hi_" + str + ".dat", ios::app);
    outfile1.setf(ios::fixed, ios::floatfield);
    outfile1.precision(6);
    outfile2.setf(ios::fixed, ios::floatfield);
    outfile2.precision(6);
    if (!outfile1.is_open())
    {
        cout << "open file failure" << endl;
    }
    for (int i = 0; i < INPUT.n_grid; i++)
    {
        outfile1 << hr[i] << endl;
        outfile2 << hi[i] << endl;
    }
    outfile1.close();
    outfile2.close();
    */


    evol1(INPUT.n_grid, hr, hi, opt.ur, opt.ui);
    if (INPUT.out_evol1 == 1)
    {
        output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_evol1_" + str + ".dat", 6, opt.ur, opt.ui);
        output_ur(INPUT.n_grid, INPUT.dir + "dl_evol1_" + str + ".dat", 6, opt.ur);
        output_ui(INPUT.n_grid, INPUT.dir + "dl_evol1_" + str + ".dat", 6, opt.ui);
    }

    /*
     ifstream ifs1("/home/xianyuer/yuer/num/tests/evol1_ur.dat");
     ifstream ifs2("/home/xianyuer/yuer/num/tests/evol1_ui.dat");
     for (int i = 0; i < INPUT.n_grid; i++)
     {
         for (int j = 0; j< INPUT.n_grid; j++)
         {
             ifs1 >> opt.ur[i][j];
             ifs2 >> opt.ui[i][j];
         }
     }
  ifs1.close();
  ifs2.close();
 */
    FFT::my_fft2d(fft, INPUT.n_grid, dk0, -2, opt.ur, opt.ui);
    if (INPUT.out_my_fft2d2 == 1)
    {
        output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_my_fft2d2_" + str + ".dat", 6, opt.ur,
                           opt.ui);
        output_ur(INPUT.n_grid, INPUT.dir + "dl_my_fft2d2_" + str + ".dat", 6, opt.ur);
        output_ui(INPUT.n_grid, INPUT.dir + "dl_my_fft2d2_" + str + ".dat", 6, opt.ui);
    }

    mdfph(INPUT.n_grid, INPUT.n1, dxyz, dlta, ddxz, -1 * wave_number, opt.ur, opt.ui);
    if (INPUT.out_mdfph2 == 1)
    {
        output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_mdfph2_" + str + ".dat", 6, opt.ur,
                           opt.ui);
        output_ur(INPUT.n_grid, INPUT.dir + "dl_mdfph2_" + str + ".dat", 6, opt.ur);
        output_ui(INPUT.n_grid, INPUT.dir + "dl_mdfph2_" + str + ".dat", 6, opt.ui);
    }
    pkkz = 1.0 / ddxz / ddxz;
    for (int i = 0; i < INPUT.n_grid; i++)
    {
        for (int j = 0; j < INPUT.n_grid; j++)
        {
            opt.ur[i][j] = opt.ur[i][j] / ddxz;
            opt.ui[i][j] = opt.ui[i][j] / ddxz;
        }
    }
    if (INPUT.out_outIntensity == 1)
    {
        output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_outIntensity_" + str + ".dat", 6, opt.ur,
                           opt.ui);
    }
}
