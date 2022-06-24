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
  a1 = 0.2391;
  default_random_engine random(a1);
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  
  int num_n = 0;
  num_n = INPUT.num_datas;
  double **a;
  a[num_n+1][105];
  a = new double *[num_n+1]();
  for (int i = 0; i < num_n+1; i++) {
    a[i] = new double[105]();
  }
  for (int i = 0; i < num_n+1; i++) {
    for (int j = 0; j < 105; j++) {
      a[i][j] = 0;
    }
  }
  ifstream ifs(
      "/home/xianyuer/yuer/num_mechinelearning/deep-light/mechinelearning/0620/data/zernike_220623_0.5_65_10000.dat");
  for (int i = 1; i <= num_n; i++) {
    for (int j = 1; j <= 65; j++) {
      ifs >> a[i][j];
    }
  }
  ifs.close();

  for (int i = 0; i < INPUT.num_datas; i++) {

    a1 = dis(random);
    OPT::Init_Phase(INPUT, opt, a1, a, i,
                    INPUT.Phase_option); // add for "random" or "confirm"

    OPT::numercial_diffraction(INPUT, a1, opt);
  }


  // time
  cout << "The run time is: " << (double)clock() / CLOCKS_PER_SEC << "s"
       << endl;
  return 0;
}
