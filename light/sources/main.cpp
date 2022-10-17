//==================================================
// Main function：The far-field transmission of focused beam is realized by fast
// Fourier transform and coordinate adaptation transform. Date: 2022-02-14
//==================================================
#include "FFt.h"
#include "Zernike.h"
#include "fun.h"
#include "input.h"
#include "optical_field.h"
#include <random>
#include <time.h>

int main() {
  // start time
  cout << "The current time is: " << (double)clock() << "s" << endl;
  Input INPUT;
  if (!INPUT.INIT(INPUT)) {
    cout << "input error!" << endl;
    exit(0);
  }
  cout << INPUT.airy << endl;
  cout << INPUT.airy * INPUT.xxz << endl;
  OPT opt;
  OPT::Init_Intensity(INPUT, opt);
  double a1 = 0.0;
  // a1 = 0.2391;

  std::uniform_real_distribution<double> dis(0.0, 1.0);
  default_random_engine random(time(0));

  opt.maxZnkDim = maxZernike(INPUT.maxZnkOrder);
  cout << "maxZnkDim=" << opt.maxZnkDim << endl;
  // cout << "INPUT.zfh * INPUT.plm / INPUT.a0 "<<INPUT.zfh * INPUT.plm /
  // INPUT.a0 << endl; cout << "INPUT.aaz "<<INPUT.aaz << endl;
  int num_n = 0;
  num_n = INPUT.num_datas;
  double **a;
  double **a0;
  // a[num_n][opt.maxZnkDim];
  a = new double *[num_n]();
  a0 = new double *[num_n]();
  for (int i = 0; i < num_n; i++) {
    a[i] = new double[opt.maxZnkDim]();
    a0[i] = new double[opt.maxZnkDim]();
  }
  for (int i = 0; i < num_n; i++) {
    for (int j = 0; j < opt.maxZnkDim; j++) {
      a[i][j] = 0;
      a0[i][j] = 0;
    }
  }

  string dir0;
  dir0 = INPUT.dir;
  // dir0 = INPUT.dir+"predict_b/b_diff_";
  // double xxzz[21] = {34.92, 35.03, 35.14, 35.24, 35.35, 35.46, 35.57,
  //                    35.68, 35.78, 35.89, 36.00, 36.11, 36.22, 36.32,
  //                    36.43, 36.54, 36.65, 36.76, 36.86, 36.97, 37.08};
  // for (int k = 10; k < 11; k++) {
  //   std::stringstream ss;
  //   ss << setprecision(4) << xxzz[k];
  //   string str;
  //   str = ss.str();
  //   string dir0;
  //   dir0 = INPUT.dir + to_string(k + 1) + "_" + str + "/diff_"; // 1_34.92_
  //   INPUT.aznk_dir = INPUT.dir + to_string(k + 1) + "_" + str +
  //   "/zernike_test_diff.txt"; cout << dir0 << endl; cout << INPUT.aznk_dir <<
  //   endl; INPUT.xxz = xxzz[k]; INPUT.aaz = INPUT.airy * INPUT.xxz; cout <<
  //   "INPUT.xxz " << INPUT.xxz << endl;

  // double defocus[21] = {-0.1,  -0.09, -0.08, -0.07, -0.06, -0.05, -0.04,
  //                       -0.03, -0.02, -0.01, 0.00,  0.01,  0.02,  0.03,
  //                       0.04,  0.05,  0.06,  0.07,  0.08,  0.09,  0.1};
  // double defocus[8] = {-0.5,  -0.4, -0.3, -0.2, 0.2, 0.3, 0.4,0.5};
  // for (int k = 0; k < 8; k++) {
  //   INPUT.aznk_dir = INPUT.dir + +"predict_b/" + to_string(k + 1) + "zernike_test_diff.txt";
    if (INPUT.Phase_option == "confirm")
    {
      ifstream ifs(INPUT.aznk_dir);
      for (int i = 0; i < num_n; i++)
      {
        for (int j = 0; j < opt.maxZnkDim - 2; j++)
        // for (int j = 0; j < opt.maxZnkDim; j++) //defocus
        {
          ifs >> a0[i][j];
          a[i][j] = a0[i][j];
        }
        // a[i][2] = a[i][2] + 1.45; //defocus
      }
      ifs.close();
    }
  // double defocus[21] = {-0.5,  -0.45, -0.4,  -0.35, -0.3, -0.25, -0.2,
  //                       -0.15, -0.1,  -0.05, 0.00,  0.05, 0.1,   0.15,
  //                       0.2,   0.25,  0.3,   0.35,  0.4,  0.45,  0.5};

  // double defocus[21] = {-0.1,  -0.09, -0.08, -0.07, -0.06, -0.05, -0.04,
  //                       -0.03, -0.02, -0.01, 0.00,  0.01,  0.02,  0.03,
  //                       0.04,  0.05,  0.06,  0.07,  0.08,  0.09,  0.1};
  
  // double defocus[8] = {-0.5,  -0.4, -0.3, -0.2, 0.2, 0.3, 0.4,
  //                       0.5};
  // for (int k = 0; k < 8; k++) {
  //   for (int i = 0; i < num_n; i++) {
  //     a[i][0] = a0[i][0] + defocus[k];
  //   }
    for (int i = 0; i < INPUT.num_datas; i++) {
      a1 = dis(random);
      OPT::Init_Phase(INPUT, opt, a1, a, i, INPUT.Phase_option,
                      dir0); // add for "random" or "confirm"
      OPT::numercial_diffraction(INPUT, a1, opt, dir0);
      cout << i << endl;
      for (int i = 0; i < INPUT.n_grid; i++) {
        delete[] opt.ur[i];
        delete[] opt.ui[i];
      }
      delete[] opt.ur;
      delete[] opt.ui;
    }
  // }

  for (int i = 0; i < num_n; i++) {
    delete[] a[i];
  }
  delete[] a;

  for (int i = 0; i < INPUT.n_grid; i++) {
    delete[] opt.ur0[i];
    delete[] opt.ui0[i];
  }
  delete[] opt.ur0;
  delete[] opt.ui0;
  // time
  cout << "The run time is: " << (double)clock() / CLOCKS_PER_SEC << "s"
       << endl;
  return 0;
}
