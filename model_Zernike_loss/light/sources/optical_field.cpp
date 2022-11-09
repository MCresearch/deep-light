#include "optical_field.h"
#include "FFt.h"
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#define PI 3.141592653589793

OPT::OPT() {}

OPT::~OPT() {}

bool OPT::Init_Intensity(Input &INPUT, OPT &opt) {
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

  opt.ur0 = new double *[INPUT.n_grid]();
  opt.ui0 = new double *[INPUT.n_grid]();

  for (int i = 0; i < INPUT.n_grid; i++) {
    opt.ur0[i] = new double[INPUT.n_grid]();
    opt.ui0[i] = new double[INPUT.n_grid]();
  }
  for (int i = 0; i < INPUT.n_grid; i++) {
    for (int j = 0; j < INPUT.n_grid; j++) {
      opt.ur0[i][j] = 0;
      opt.ui0[i][j] = 0;
    }
  }

  // set inIntensity
  a02 = INPUT.a0 * INPUT.a0;
  for (int i = 0; i < INPUT.n_grid; i++) {
    x = (i + 1 - INPUT.n1) * INPUT.dxy0; // +不加1
    x2 = x * x;
    for (int j = 0; j < INPUT.n_grid; j++) {
      y = (j + 1 - INPUT.n1) * INPUT.dxy0;
      y2 = y * y;
      r2 = x2 + y2;
      if (r2 <= a02) {
        opt.ur0[i][j] = exp(-1 * pow(r2 / a02, INPUT.mgs));
        opt.ui0[i][j] = 0;
      }
    }
  }
  // save inIntensity
  if (INPUT.out_inIntensity == 1) {
    output_inIntensity(INPUT.n_grid, INPUT.dir + "dl_inIntensity.dat", 6,
                       opt.ur0, opt.ui0);
    // output_ur(INPUT.n_grid, dir0 + "dl_inIntensity.dat", 6, opt.ur0);
    // output_ui(INPUT.n_grid, dir0 + "dl_inIntensity.dat", 6, opt.ui0);
  }
  return true;
}

bool OPT::Init_Phase(Input &INPUT, OPT &opt, double a1, double **a, int num,
                     const string type, string dir0) {

  opt.ur = new double *[INPUT.n_grid]();
  opt.ui = new double *[INPUT.n_grid]();

  for (int i = 0; i < INPUT.n_grid; i++) {
    opt.ur[i] = new double[INPUT.n_grid]();
    opt.ui[i] = new double[INPUT.n_grid]();
  }
  for (int i = 0; i < INPUT.n_grid; i++) {
    for (int j = 0; j < INPUT.n_grid; j++) {
      opt.ur[i][j] = 0;
      opt.ui[i][j] = 0;
    }
  }
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
  double a02 = 0.0;

  char buffer[20];
  sprintf(buffer, "%f", a1);
  string str = buffer;
  a02 = INPUT.a0 * INPUT.a0;
  opt.nznk = new int[opt.maxZnkDim + 1]();
  opt.mznk = new int[opt.maxZnkDim + 1]();
  // nmlznk(INPUT.maxZnkOrder, opt.maxZnkDim, opt.nznk, opt.mznk);  // delete
  // lznk?
  mnznk(INPUT.maxZnkOrder, opt.maxZnkDim, opt.nznk, opt.mznk);
  opt.aznk = new double[opt.maxZnkDim + 1]();
  opt.eznk = new double[opt.maxZnkDim + 1]();
  opt.ph = new double *[INPUT.n_grid]();
  opt.pl = new double[opt.maxZnkDim + 1]();

  for (int i = 0; i <= opt.maxZnkDim; i++) {
    opt.pl[i] = 0;
  }

  for (int i = 0; i < INPUT.n_grid; i++) {
    opt.ph[i] = new double[INPUT.n_grid]();
  }

  if (type == "random") {
    double rdmg = 0;
    // cout << a1 << endl;
    // default_random_engine            random(a1);
    // std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 3; i <= opt.maxZnkDim; i++) {
      rdm_gauss(a1, rdmg);
      // rdmg = dis(random);
      // cout << i << "\t" << rdmg << endl;
      opt.eznk[i] = exp(-opt.nznk[i] * INPUT.eeznk);
      opt.aznk[i] = rdmg * opt.eznk[i];
      ss = ss + pow(opt.aznk[i], 2);
    }

    //系数按方差rms归一化
    for (int i = 3; i <= opt.maxZnkDim; i++) {
      opt.aznk[i] = opt.aznk[i] * sqrt(INPUT.rms / ss);
    }
    if (INPUT.out_zernike_coeff == 1) {
      // output_zernike_coeff(INPUT.n_grid,
      //                      dir0 + "dl_zernike_coeff_" + str + ".dat", 7,
      //                      opt.maxZnkDim, opt.aznk, opt.nznk, opt.eznk);
      output_zernike_coeff_0(INPUT.n_grid, dir0 + "dl_zernike_coeff.dat", 7,
                             opt.maxZnkDim, opt.aznk, opt.nznk, opt.eznk);
    }
    // opt.aznk[1] = 0;
    // opt.aznk[2] = 0;
    // opt.aznk[3] = 0.289685920360869 ;
    // opt.aznk[4] = 1.34470382347105;
    // opt.aznk[5] = -0.133870705620043;
    // opt.aznk[6] = 0.541865557255511;
    // opt.aznk[7] = 0.386754435348797;
    // opt.aznk[8] = 1.36843853446997;
    // opt.aznk[9] = -0.711775011298688;

    // for (int j = 1; j <= opt.maxZnkDim; j++) {
    //   opt.aznk[j] = a[num][j - 1];
    //   cout << j << " " << opt.aznk[j] << endl;
    // }

    // for (int i = 3; i <= opt.maxZnkDim; i++) {
    //   opt.eznk[i] = exp(-opt.nznk[i] * INPUT.eeznk);
    //   opt.aznk[i] = opt.aznk[i] * opt.eznk[i];
    //   // ss = ss + pow(opt.aznk[i], 2);
    // }
    // //系数按方差rms归一化
    // for (int i = 3; i <= opt.maxZnkDim; i++) {
    //   opt.aznk[i] = opt.aznk[i] * sqrt(INPUT.rms / ss);
    // }
    for (int i = 0; i < INPUT.n_grid; i++) {
      x = (i + 1 - INPUT.n1) * INPUT.dxy0;
      x2 = x * x;
      for (int j = 0; j < INPUT.n_grid; j++) {
        y = (j + 1 - INPUT.n1) * INPUT.dxy0;
        y2 = y * y;
        r2 = x2 + y2;
        // cout << i <<" "<< j <<" "<< endl;
        if (r2 / a02 <= 1) {
          zernike_cg(opt.maxZnkDim, x / INPUT.a0, y / INPUT.a0, opt.pl);
          // radial_polynomials(opt.maxZnkDim, x, y, opt.nznk, opt.mznk,
          // opt.pl);
          // cout << i <<" "<< j <<" "<< endl;
          opt.ph[i][j] = 0;
          for (int l = INPUT.minZnkDim; l <= opt.maxZnkDim; l++) {
            opt.ph[i][j] = opt.ph[i][j] + opt.pl[l] * opt.aznk[l];
            // cout << l << "\t" << opt.pl[l] << "\t" << opt.aznk[l] << endl;
          }
          // cout << i<<" "<<j<<" "<<"12221" << endl;
          uri = opt.ur0[i][j];
          opt.ur[i][j] = uri * cos(opt.ph[i][j]);
          opt.ui[i][j] = uri * sin(opt.ph[i][j]);
          // cout << i<<" "<<j<<" "<< "12222" << endl;
        }
      }
    }
    // outfile23.close();
    // output_ur(INPUT.n_grid, dir0 + "dl_ph_.dat", 6, opt.ph);
  } else if (type == "confirm") {
    for (int j = 3; j <= opt.maxZnkDim; j++) {
      opt.aznk[j] = a[num][j - 3];
      // opt.aznk[j] = a[num][j-1]; //defocus
      // cout << j << " " << opt.aznk[j] << endl;
    }
    // for (int i = 3; i <= opt.maxZnkDim; i++) {
    //   opt.eznk[i] = exp(-opt.nznk[i] * INPUT.eeznk);
    //   opt.aznk[i] = opt.aznk[i] * opt.eznk[i];
    //   //ss = ss + pow(opt.aznk[i], 2);
    // }
    //系数按方差rms归一化
    // for (int i = 3; i <= opt.maxZnkDim; i++) {
    //   opt.aznk[i] = opt.aznk[i] * sqrt(INPUT.rms / ss);
    // }
    if (INPUT.out_zernike_coeff == 1) {
      // output_zernike_coeff(INPUT.n_grid,
      //                      dir0 + "dl_zernike_coeff_" + str + ".dat", 7,
      //                      opt.maxZnkDim, opt.aznk, opt.nznk, opt.eznk);
      output_zernike_coeff_0(INPUT.n_grid, dir0 + "dl_zernike_coeff.dat", 7,
                             opt.maxZnkDim, opt.aznk, opt.nznk, opt.eznk);
    }
    for (int i = 0; i < INPUT.n_grid; i++) {
      x = (i + 1 - INPUT.n1) * INPUT.dxy0;
      x2 = x * x;
      for (int j = 0; j < INPUT.n_grid; j++) {
        y = (j + 1 - INPUT.n1) * INPUT.dxy0;
        y2 = y * y;
        r2 = x2 + y2;
        // cout << i <<" "<< j <<" "<< endl;
        if (r2 / a02 <= 1) {
          zernike_cg(opt.maxZnkDim, x / INPUT.a0, y / INPUT.a0, opt.pl);
          // radial_polynomials(opt.maxZnkDim, x, y, opt.nznk, opt.mznk,
          // opt.pl);
          // cout << i <<" "<< j <<" "<< endl;
          opt.ph[i][j] = 0;
          for (int l = INPUT.minZnkDim; l <= opt.maxZnkDim; l++) {
            opt.ph[i][j] = opt.ph[i][j] + opt.pl[l] * opt.aznk[l];
            // cout << l << "\t" << opt.pl[l] << "\t" << opt.aznk[l] << endl;
          }
          // cout << i<<" "<<j<<" "<<"12221" << endl;
          uri = opt.ur0[i][j];
          opt.ur[i][j] = uri * cos(opt.ph[i][j]);
          opt.ui[i][j] = uri * sin(opt.ph[i][j]);
          // cout << i<<" "<<j<<" "<< "12222" << endl;
        }
      }
    }
    // output_ur(INPUT.n_grid, dir0 + "dl_ph_.dat", 6, opt.ph);
  } else {
    cout << "input phase type error!" << endl;
    exit(0);
  }
  if (INPUT.out_inPhase == 1) {
    output_inIntensity(INPUT.n_grid, dir0 + "dl_inPhase_" + str + ".dat", 6,
                       opt.ur, opt.ui);
    // output_inPhase(INPUT.n_grid, dir0 + "dl_inPhase_" + str + ".dat", 6,
    // opt.ph);
    output_ur(INPUT.n_grid, dir0 + "dl_inPhase_" + str + ".dat", 6, opt.ur);
    output_ui(INPUT.n_grid, dir0 + "dl_inPhase_" + str + ".dat", 6, opt.ui);
  }

  delete[] opt.nznk;
  delete[] opt.mznk;
  delete[] opt.aznk;
  delete[] opt.eznk;
  delete[] opt.pl;

  for (int i = 0; i < INPUT.n_grid; i++) {
    delete[] opt.ph[i];
  }
  delete[] opt.ph;

  return true;
}

void OPT::numercial_diffraction(Input &INPUT, const double a1, OPT &opt,
                                string dir0) {
  cout << "numercial_diffraction" << endl;
  double dxy0 = 0.0;
  double dxyz = 0.0;
  double dlta = 0.0;
  double ddxz = 0.0;
  double dk0 = 0.0;
  double zzzz = 0.0;
  double wave_number = 0.0;
  double pkkz = 0.0;
  double *hr;
  double *hi;
  FFT fft;

  char buffer[20];
  sprintf(buffer, "%f", a1);
  string str = buffer;

  hr = new double[INPUT.n_grid]();
  hi = new double[INPUT.n_grid]();

  FFT::fft_initialize(INPUT.mm, INPUT.n_grid, fft);

  // output_inIntensity(INPUT.n_grid, dir0 + "dl_fft_initialize.dat", 6,
  // opt.ur, opt.ui);

  dxy0 = INPUT.aa0 / INPUT.n_grid;
  dxyz = INPUT.aaz / INPUT.n_grid;

  dlta = (1 - INPUT.aaz / INPUT.aa0) / INPUT.zfh;
  ddxz = 1 - dlta * INPUT.zfh;

  dk0 = 1 / INPUT.aa0;
  zzzz = INPUT.zfh / (1 - dlta * INPUT.zfh);
  wave_number = 2 * PI / INPUT.plm;

  // output_ur(INPUT.n_grid, dir0 + "dl_ur_" + str + ".dat", 6, opt.ur);
  // output_ur(INPUT.n_grid, dir0 + "dl_ui_" + str + ".dat", 6, opt.ui);

  focusing(INPUT.n_grid, INPUT.n1, wave_number, dxy0, 1 / INPUT.zfh, opt.ur,
           opt.ui);
  if (INPUT.out_focusing == 1) {
    output_inIntensity(INPUT.n_grid, dir0 + "dl_focusing_" + str + ".dat", 6,
                       opt.ur, opt.ui);
    output_ur(INPUT.n_grid, dir0 + "dl_focusing_" + str + ".dat", 6, opt.ur);
    output_ui(INPUT.n_grid, dir0 + "dl_focusing_" + str + ".dat", 6, opt.ui);
  }
  mdfph(INPUT.n_grid, INPUT.n1, dxy0, dlta, 1, wave_number, opt.ur, opt.ui);
  if (INPUT.out_mdfph1 == 1) {
    output_inIntensity(INPUT.n_grid, dir0 + "dl_mdfph1_" + str + ".dat", 6,
                       opt.ur, opt.ui);
    // cout << "dxy0" << dxy0 << "dlta" << dlta << "wave_number" << wave_number
    // << endl;
    output_ur(INPUT.n_grid, dir0 + "dl_mdfph1_" + str + ".dat", 6, opt.ur);
    output_ui(INPUT.n_grid, dir0 + "dl_mdfph1_" + str + ".dat", 6, opt.ui);
  }
  FFT::my_fft2d(fft, INPUT.n_grid, dxy0, 2, opt.ur, opt.ui);
  if (INPUT.out_my_fft2d1 == 1) {
    output_inIntensity(INPUT.n_grid, dir0 + "dl_my_fft2d1_" + str + ".dat", 6,
                       opt.ur, opt.ui);
    output_ur(INPUT.n_grid, dir0 + "dl_my_fft2d1_" + str + ".dat", 6, opt.ur);
    output_ui(INPUT.n_grid, dir0 + "dl_my_fft2d1_" + str + ".dat", 6, opt.ui);
  }

  prop1(INPUT.n_grid, INPUT.n1, zzzz, wave_number, INPUT.aa0, hr, hi);

  // ofstream outfile1;
  // ofstream outfile2;
  // outfile1.open(dir0 + "dl_prop1_hr.dat", ios::app);
  // outfile2.open(dir0 + "dl_prop1_hi.dat", ios::app);
  // outfile1.setf(ios::fixed, ios::floatfield);
  // outfile1.precision(6);
  // outfile2.setf(ios::fixed, ios::floatfield);
  // outfile2.precision(6);
  // if (!outfile1.is_open())
  // {
  //     cout << "open file failure" << endl;
  // }
  // for (int i = 0; i < INPUT.n_grid; i++)
  // {
  //     outfile1 << hr[i] << endl;
  //     outfile2 << hi[i] << endl;
  // }
  // outfile1.close();
  // outfile2.close();

  evol1(INPUT.n_grid, hr, hi, opt.ur, opt.ui);
  if (INPUT.out_evol1 == 1) {
    output_inIntensity(INPUT.n_grid, dir0 + "dl_evol1_" + str + ".dat", 6,
                       opt.ur, opt.ui);
    output_ur(INPUT.n_grid, dir0 + "dl_evol1_" + str + ".dat", 6, opt.ur);
    output_ui(INPUT.n_grid, dir0 + "dl_evol1_" + str + ".dat", 6, opt.ui);
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
  if (INPUT.out_my_fft2d2 == 1) {
    output_inIntensity(INPUT.n_grid, dir0 + "dl_my_fft2d2_" + str + ".dat", 6,
                       opt.ur, opt.ui);
    output_ur(INPUT.n_grid, dir0 + "dl_my_fft2d2_" + str + ".dat", 6, opt.ur);
    output_ui(INPUT.n_grid, dir0 + "dl_my_fft2d2_" + str + ".dat", 6, opt.ui);
  }

  delete[] fft.kk;
  delete[] fft.kj;
  delete[] fft.km0;
  delete[] fft.wr;
  delete[] fft.wi;
  for (int i = 0; i <= INPUT.n_grid; i++) {
    //??? fortran km的第二个下标是从1开始的现在为0
    for (int j = 0; j < INPUT.mm; j++) {
      delete[] fft.km[i][j]; //??? fortran km的第三个下标是从1开始的现在为0
    }
    delete[] fft.km[i];
  }
  delete[] fft.km;

  mdfph(INPUT.n_grid, INPUT.n1, dxyz, dlta, ddxz, -1 * wave_number, opt.ur,
        opt.ui);
  if (INPUT.out_mdfph2 == 1) {
    output_inIntensity(INPUT.n_grid, dir0 + "dl_mdfph2_" + str + ".dat", 6,
                       opt.ur, opt.ui);
    output_ur(INPUT.n_grid, dir0 + "dl_mdfph2_" + str + ".dat", 6, opt.ur);
    output_ui(INPUT.n_grid, dir0 + "dl_mdfph2_" + str + ".dat", 6, opt.ui);
  }

  // pkkz = 1.0 / ddxz / ddxz;
  // cout << ddxz << endl;
  double **intensity;
  double **down_intensity;
  intensity = new double *[INPUT.n_grid];
  down_intensity = new double *[128];
  for (int i = 0; i < INPUT.n_grid; i++) {
    intensity[i] = new double[INPUT.n_grid]();
  }
  for (int i = 0; i < 128; i++) {
    down_intensity[i] = new double[128]();
  }
  for (int i = 0; i < INPUT.n_grid; i++) {
    for (int j = 0; j < INPUT.n_grid; j++) {
      opt.ur[i][j] = opt.ur[i][j] / ddxz;
      opt.ui[i][j] = opt.ui[i][j] / ddxz;
      intensity[i][j] = pow(opt.ur[i][j], 2) + pow(opt.ui[i][j], 2);
    }
  }

  if (INPUT.out_outIntensity == 1) {
    output_inIntensity(INPUT.n_grid, dir0 + "dl_outIntensity.dat", 1, opt.ur,
                       opt.ui);
  }

  double max = 0;
  for (int i = 0; i < INPUT.n_grid; i = i + 2) {
    for (int j = 0; j < INPUT.n_grid; j = j + 2) {

      max = intensity[i][j];
      if (max < intensity[i][j + 1]) {
        max = intensity[i][j + 1];
      }
      if (max < intensity[i + 1][j]) {
        max = intensity[i + 1][j];
      }
      if (max < intensity[i + 1][j + 1]) {
        max = intensity[i + 1][j + 1];
      }
      down_intensity[i / 2][j / 2] = max;
    }
  }
  output_ur(INPUT.n_grid/2, dir0 + "dl_down_intensity.dat", 1, down_intensity);

  delete[] hr;
  delete[] hi;
  delete[] down_intensity;
  delete[] intensity;
}
