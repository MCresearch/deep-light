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
  cout << "aaa" << recv(17) << endl;
  if (!INPUT.INIT(INPUT)) {
    cout << "input error!" << endl;
    exit(0);
  }

  OPT opt;
  OPT::Init_Intensity(INPUT, opt);

  double a1 = 0.0;
  // a1 = 0.2391;
  // default_random_engine random(a1);
  // std::uniform_real_distribution<double> dis(0.0, 1.0);
  opt.maxZnkDim = maxZernike(INPUT.maxZnkOrder);
  cout << "maxZnkDim=" << opt.maxZnkDim << endl;

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
  ifstream ifs(INPUT.aznk_dir);
  for (int i = 0; i < num_n; i++) {
    for (int j = 0; j < opt.maxZnkDim; j++) {
      ifs >> a[i][j];
    }
  }
  ifs.close();

  for (int i = 0; i < INPUT.num_datas; i++) {

    // a1 = dis(random);
    OPT::Init_Phase(INPUT, opt, a1, a, i,
                    INPUT.Phase_option); // add for "random" or "confirm"

    OPT::numercial_diffraction(INPUT, a1, opt);
  }


  // time
  cout << "The run time is: " << (double)clock() / CLOCKS_PER_SEC << "s"
       << endl;
  return 0;
}
