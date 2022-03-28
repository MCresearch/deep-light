#include "optical_field.h"
#include <random>

OPT::OPT() {}

OPT::~OPT() {}

bool OPT::Init(OPT& opt)
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

    for (int i = 0; i < INPUT.n_grid; i++)
    {
        opt.ur[i] = new double[INPUT.n_grid]();
    }
    for (int i = 0; i < INPUT.n_grid; i++)
    {
        opt.ui[i] = new double[INPUT.n_grid]();
    }


    // 设置光强
    a02 = INPUT.a0 * INPUT.a0;
    for (int j = 0; j < INPUT.n_grid; j++)
    {
        y = (j - INPUT.n1) * INPUT.dxy0;
        y2 = y * y;
        for (int i = 0; i < INPUT.n_grid; i++)
        {
            x = (i - INPUT.n1) * INPUT.dxy0;
            x2 = x * x;
            r2 = x2 + y2;
            opt.ur[i][j] = exp(-1 * pow(r2 / a02, INPUT.mgs));
            opt.ui[i][j] = 0;
        }
    }
    // 存储光强

    // 相位
    opt.maxZnkDim = maxZernike(INPUT.maxZnkOrder);
    cout << "maxZnkDim =" << opt.maxZnkDim << endl;
    opt.nznk = new int[opt.maxZnkDim]();
    opt.mznk = new int[opt.maxZnkDim]();
    opt.lznk = new int[opt.maxZnkDim]();
    nmlznk(INPUT.maxZnkOrder, opt.maxZnkDim, opt.nznk, opt.mznk, opt.lznk);

    // 多项式系数：高斯随机数
    a1 = 0.2391;  //随机数初值
    ss = 0;

    opt.aznk = new double[opt.maxZnkDim]();
    opt.eznk = new double[opt.maxZnkDim]();
    opt.ph = new double*[INPUT.n_grid]();
    opt.pl = new double[opt.maxZnkDim]();
    for (int i = 0; i < INPUT.n_grid; i++)
    {
        opt.ph[i] = new double[INPUT.n_grid]();
    }


    default_random_engine            random(a1);
    std::normal_distribution<double> dis(0, 1);

    for (int i = 3; i <= opt.maxZnkDim; i++)
    {
        rdmg = dis(random);
        opt.eznk[i] = exp(-opt.nznk[i] * INPUT.eeznk);
        opt.aznk[i] = rdmg * opt.eznk[i];
        ss = ss + pow(opt.aznk[i], 2);
    }

    //系数按方差rms归一化
    for (int i = 3; i <= opt.maxZnkDim; i++)
    {
        opt.aznk[i] = opt.aznk[i] * sqrt(INPUT.rms / ss);
    }
    /*
    open(11,file='zernike_coeff.dat')
    do i=1,maxZnkDim
    write(11,*)i,aznk(i),nznk(i),eznk(i)
    end do
    close(1)
    */

    //设置相位
    for (int j = 0; j < INPUT.n_grid; j++)
    {
        y = (j - INPUT.n1) * INPUT.dxy0;
        y2 = y * y;
        for (int i = 0; i < INPUT.n_grid; i++)
        {
            x = (i - INPUT.n1) * INPUT.dxy0;
            x2 = x * x;
            r2 = x2 + y2;
            if (r2 / a02 <= 1)
            {
                zernike_cg(opt.maxZnkDim, opt.pl, x / INPUT.a0, y / INPUT.a0);
                opt.ph[i][j] = 0;
                for (int l = INPUT.minZnkDim; l <= opt.maxZnkDim; l++)
                {
                    opt.ph[i][j] = opt.ph[i][j] + opt.pl[l] * opt.aznk[l];
                }
                uri = opt.ur[i][j];
                opt.ur[i][j] = uri * cos(opt.ph[i][j]);
                opt.ui[i][j] = uri * sin(opt.ph[i][j]);
            }
        }
    }

    /*
open(1,file='inPhase.dat')
do j=1,n_grid
write(1,111)(ph(i,j),i=1,n_grid)
end do
close(1)
*/
}

void OPT::numercial_diffraction(OPT& opt)
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

    for (int i = 0; i < INPUT.n_grid; i++)
    {
        opt.ur[i] = new double[INPUT.n_grid]();
    }

    dxy0 = INPUT.aa0 / INPUT.n_grid;
    dxyz = INPUT.aaz / INPUT.n_grid;

    dlta = (1 - INPUT.aaz / INPUT.aa0) / INPUT.zfh;
    ddxz = 1 - dlta * INPUT.zfh;

    dk0 = 1 / INPUT.aa0;
    zzzz = INPUT.zfh / (1 - dlta * INPUT.zfh);
    PI = atan(1) * 4;
    wave_number = 2 * PI / INPUT.plm;

    focusing(INPUT.n_grid, INPUT.n9, INPUT.n1, opt.ur, opt.ui, wave_number, dxy0, 1 / INPUT.zfh);

    mdfph(INPUT.n_grid, INPUT.n9, INPUT.n1, opt.ur, opt.ui, dxy0, dlta, 1, wave_number);

    my_fft2d(fft, INPUT.n9, opt.ur, opt.ui, dxy0, 2);

    prop1(INPUT.n_grid, INPUT.n9, INPUT.n1, hr, hi, zzzz, wave_number, INPUT.aa0);

    evol1(INPUT.n_grid, INPUT.n9, hr, hi, opt.ur, opt.ui);

    my_fft2d(fft, INPUT.n9, opt.ur, opt.ui, dk0, -2);

    mdfph(INPUT.n_grid, INPUT.n9, INPUT.n1, opt.ur, opt.ui, dxyz, dlta, ddxz, -wave_number);

    pkkz = 1.0 / ddxz / ddxz;
    for (int j = 0; j < INPUT.n_grid; j++)
    {
        for (int i = 0; i < INPUT.n_grid; i++)
        {
            ur[i][j] = ur[i][j] / ddxz;
            ui[i][j] = ui[i][j] / ddxz;
        }
    }
}
