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
  // a[num_n][opt.maxZnkDim];
  a = new double *[num_n]();
  for (int i = 0; i < num_n; i++) {
    a[i] = new double[opt.maxZnkDim]();
  }
  for (int i = 0; i < num_n; i++) {
    for (int j = 0; j < opt.maxZnkDim; j++) {
      a[i][j] = 0;
    }
  }
  if (INPUT.Phase_option == "confirm") {
    ifstream ifs(INPUT.aznk_dir);
    for (int i = 0; i < num_n; i++) {
      for (int j = 0; j < opt.maxZnkDim - 2; j++) {
        ifs >> a[i][j];
      }
    }
    ifs.close();
  }
  for (int i = 0; i < INPUT.num_datas; i++) {
    a1 = dis(random);
    OPT::Init_Phase(INPUT, opt, a1, a, i,
                    INPUT.Phase_option); // add for "random" or "confirm"
    OPT::numercial_diffraction(INPUT, a1, opt);
    cout << i << endl;

    for (int i = 0; i <INPUT.n_grid; i++) {
      delete[] opt.ur[i];
      delete[] opt.ui[i];
    }
    delete[] opt.ur;
    delete[] opt.ui;
  }



  for (int i = 0; i < num_n; i++) {
    delete[] a[i];
  }
  delete[] a;

    for (int i = 0; i <INPUT.n_grid; i++) {
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
