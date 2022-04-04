#include "optical_field.h"
#include "FFt.h"
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
    for (int j = 0; j < INPUT.n_grid; j++)
    {
        y = (j + 1 - INPUT.n1) * INPUT.dxy0;  // +不加1
        y2 = y * y;
        for (int i = 0; i < INPUT.n_grid; i++)
        {
            x = (i + 1 - INPUT.n1) * INPUT.dxy0;
            x2 = x * x;
            r2 = x2 + y2;
            opt.ur[i][j] = exp(-1 * pow(r2 / a02, INPUT.mgs));
            opt.ui[i][j] = 0;
        }
    }
    // save inIntensity
    output_inIntensity(INPUT.n_grid, "./tests/dl_inIntensity.dat", 6, opt.ur, opt.ui);
}


bool OPT::Init_Phase(Input &INPUT, OPT &opt, const double a1, const string type)
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
    ss = 0;
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

        double   rdmg[200];
        ifstream ifs("rdmg.dat");
        for (int i = 3; i <= opt.maxZnkDim; i++)
        {
            ifs >> rdmg[i];
        }
        ifs.close();

        for (int i = 3; i <= opt.maxZnkDim; i++)
        {
            // rdmg = dis(random);
            // rdm_gauss(a1,rdmg);
            // cout << i << "\t"<<rdmg[i] << endl;
            opt.eznk[i] = exp(-opt.nznk[i] * INPUT.eeznk);
            opt.aznk[i] = rdmg[i] * opt.eznk[i];
            ss = ss + pow(opt.aznk[i], 2);
        }

        //系数按方差rms归一化
        for (int i = 3; i <= opt.maxZnkDim; i++)
        {
            opt.aznk[i] = opt.aznk[i] * sqrt(INPUT.rms / ss);
        }

        output_zernike_coeff(INPUT.n_grid, "./tests/dl_zernike_coeff.dat", 7, opt.maxZnkDim,
                             opt.aznk, opt.nznk, opt.eznk);

        ofstream outfile23;
        outfile23.open("./tests/dl_pl_aznk.dat", ios::app);

        //设置相位
        for (int j = 0; j < INPUT.n_grid; j++)
        {
            y = (j + 1 - INPUT.n1) * INPUT.dxy0;
            y2 = y * y;
            for (int i = 0; i < INPUT.n_grid; i++)
            {
                x = (i + 1 - INPUT.n1) * INPUT.dxy0;
                x2 = x * x;
                r2 = x2 + y2;
                if (r2 / a02 <= 1)
                {
                    zernike_cg(opt.maxZnkDim, x / INPUT.a0, y / INPUT.a0, opt.pl);

                    ofstream outfile22;
                    outfile22.open("./tests/dl_zernike_cg.dat", ios::app);
                    // outfile22.open("./tests/dl_x_y.dat", ios::app);
                    outfile22.setf(ios::fixed, ios::floatfield);
                    outfile22.precision(7);
                    // outfile22 << i << "\t" << j << "\t" <<  x/INPUT.a0 << "\t" <<  y/INPUT.a0<<
                    // endl;

                    if (!outfile22.is_open())
                    {
                        cout << "./tests/open file failure" << endl;
                    }
                    for (int i = 1; i <= opt.maxZnkDim; i++)
                    {
                        outfile22 << i << "\t" << opt.pl[i] << endl;
                    }
                    outfile22.close();

                    opt.ph[i][j] = 0;
                    for (int l = INPUT.minZnkDim; l <= opt.maxZnkDim; l++)
                    {
                        opt.ph[i][j] = opt.ph[i][j] + opt.pl[l] * opt.aznk[l];
                        outfile23 << l << "\t" << opt.pl[l] << "\t" << opt.aznk[l] << endl;
                    }
                    uri = opt.ur[i][j];
                    opt.ur[i][j] = uri * cos(opt.ph[i][j]);
                    opt.ui[i][j] = uri * sin(opt.ph[i][j]);
                }
            }
        }
        outfile23.close();
    }

    else if (type == "confirm")
    {
        ifstream ifs1("./tests/inPhase.dat");
        for (int i = 0; i < INPUT.n_grid; i++)
        {
            for (int j = 0; j < INPUT.n_grid; j++)
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

    output_inIntensity(INPUT.n_grid, "./tests/dl_inPhase_intensity.dat", 6, opt.ur, opt.ui);
    output_inPhase(INPUT.n_grid, "./tests/dl_inPhase.dat", 6, opt.ph);

    delete[] opt.aznk;
    delete[] opt.eznk;
    delete[] opt.pl;
    for (int i = 0; i < opt.maxZnkDim; i++)
        delete[] opt.ph[i];
    delete[] opt.ph;
}


void OPT::numercial_diffraction(Input &INPUT, OPT &opt)
{
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
    hr = new double[INPUT.n_grid]();
    hi = new double[INPUT.n_grid]();

    FFT::fft_initialize(INPUT.mm, INPUT.n_grid, fft);
    output_inIntensity(INPUT.n_grid, "./tests/dl_fft_initialize.dat", 6, opt.ur, opt.ui);

    dxy0 = INPUT.aa0 / INPUT.n_grid;
    dxyz = INPUT.aaz / INPUT.n_grid;

    dlta = (1 - INPUT.aaz / INPUT.aa0) / INPUT.zfh;
    ddxz = 1 - dlta * INPUT.zfh;

    dk0 = 1 / INPUT.aa0;
    zzzz = INPUT.zfh / (1 - dlta * INPUT.zfh);
    wave_number = 2 * PI / INPUT.plm;



    focusing(INPUT.n_grid, INPUT.n1, wave_number, dxy0, 1 / INPUT.zfh, opt.ur, opt.ui);
    output_inIntensity(INPUT.n_grid, "./tests/dl_focusing.dat", 6, opt.ur, opt.ui);

    mdfph(INPUT.n_grid, INPUT.n1, dxy0, dlta, 1, wave_number, opt.ur, opt.ui);
    output_inIntensity(INPUT.n_grid, "./tests/dl_mdfph1.dat", 6, opt.ur, opt.ui);
    // cout << "dxy0" << dxy0 << "dlta" << dlta << "wave_number" << wave_number << endl;
    output_ur(INPUT.n_grid, "./tests/dl_mdfph1_ur.dat", 6, opt.ur);

    FFT::my_fft2d(fft, INPUT.n_grid, dxy0, 2, opt.ur, opt.ui);
    output_inIntensity(INPUT.n_grid, "./tests/dl_my_fft2d1.dat", 6, opt.ur, opt.ui);
    output_ur(INPUT.n_grid, "./tests/dl_my_fft2d1_ur.dat", 6, opt.ur);
    output_ur(INPUT.n_grid, "./tests/dl_my_fft2d1_ui.dat", 6, opt.ui);


    prop1(INPUT.n_grid, INPUT.n1, zzzz, wave_number, INPUT.aa0, hr, hi);
    output_inIntensity(INPUT.n_grid, "./tests/dl_prop1.dat", 6, opt.ur, opt.ui);


    evol1(INPUT.n_grid, hr, hi, opt.ur, opt.ui);
    output_inIntensity(INPUT.n_grid, "./tests/dl_evol1.dat", 6, opt.ur, opt.ui);


    FFT::my_fft2d(fft, INPUT.n_grid, dk0, -2, opt.ur, opt.ui);
    output_inIntensity(INPUT.n_grid, "./tests/dl_my_fft2d2.dat", 6, opt.ur, opt.ui);
    // cout << "dxy0" << dxy0<< endl;
    // cout << "dk0" << dk0 << endl;


    mdfph(INPUT.n_grid, INPUT.n1, dxyz, dlta, ddxz, -1 * wave_number, opt.ur, opt.ui);
    output_inIntensity(INPUT.n_grid, "./tests/dl_mdfph2.dat", 6, opt.ur, opt.ui);

    pkkz = 1.0 / ddxz / ddxz;
    for (int j = 0; j < INPUT.n_grid; j++)
    {
        for (int i = 0; i < INPUT.n_grid; i++)
        {
            opt.ur[i][j] = opt.ur[i][j] / ddxz;
            opt.ui[i][j] = opt.ui[i][j] / ddxz;
        }
    }

    output_inIntensity(INPUT.n_grid, "./tests/dl_outIntensity.dat", 6, opt.ur, opt.ui);
}
