#include "../../light/optical_field.h"

#include <limits.h>

#include "gtest/gtest.h"

namespace {
Input INPUT;
OPT opt;
TEST(optical_field, Init_Intensity) {

  EXPECT_TRUE(INPUT.INIT(INPUT));
  EXPECT_TRUE(OPT::Init_Intensity(INPUT, opt));

  ifstream ifs1("inIntensity_ur_32.dat");
  ifstream ifs2("inIntensity_ui_32.dat");
  double **ur;
  double **ui;
  ur = new double *[INPUT.n_grid]();
  ui = new double *[INPUT.n_grid]();
  int t = 0;
  double x = 0;
  for (int i = 0; i < INPUT.n_grid; i++) {
    ur[i] = new double[INPUT.n_grid]();
    ui[i] = new double[INPUT.n_grid]();
    for (int j = 0; j < INPUT.n_grid; j++) {
      ur[i][j] = 0;
      ui[i][j] = 0;
      ifs1 >> ur[i][j];
      ifs2 >> ui[i][j];
      /*
      opt.ur0[i][j] = opt.ur0[i][j] *
      10000000;//因为要保留最后2位数，且是对最后一位进行四舍五入就是操作最后3位数字；就放大1000
            t =
      (opt.ur0[i][j]+5)/10;//最后一位数+5，再取整；把多余的小数位通过取整舍去；
            opt.ur0[i][j] = (double)t/1000000; //别忘了把t的类型转化一下；
      opt.ui0[i][j] = opt.ui0[i][j] *
      10000000;//因为要保留最后2位数，且是对最后一位进行四舍五入就是操作最后3位数字；就放大1000
            t =
      (opt.ui0[i][j]+5)/10;//最后一位数+5，再取整；把多余的小数位通过取整舍去；
            opt.ui0[i][j] = (double)t/1000000; //别忘了把t的类型转化一下；
      */
    }
  }

  for (int i = 0; i < INPUT.n_grid; i++) {
    for (int j = 0; j < INPUT.n_grid; j++) {
      EXPECT_NEAR(ur[i][j], opt.ur0[i][j], 1e5);
      EXPECT_NEAR(ui[i][j], opt.ui0[i][j], 1e5);
      /*
      EXPECT_DOUBLE_EQ(ur[i][j], opt.ur0[i][j]);
      EXPECT_DOUBLE_EQ(ui[i][j], opt.ui0[i][j]);
      */
    }
  }
}

TEST(optical_field, Init_Phase_confirm) {

  EXPECT_TRUE(OPT::Init_Phase(INPUT, opt,0.2391,"confirm"));

  ifstream ifs1("inPhase_ur_32.dat");
  ifstream ifs2("inPhase_ui_32.dat");
  double **ur;
  double **ui;
  ur = new double *[INPUT.n_grid]();
  ui = new double *[INPUT.n_grid]();
  int t = 0;
  double x = 0;
  for (int i = 0; i < INPUT.n_grid; i++) {
    ur[i] = new double[INPUT.n_grid]();
    ui[i] = new double[INPUT.n_grid]();
    for (int j = 0; j < INPUT.n_grid; j++) {
      ur[i][j] = 0;
      ui[i][j] = 0;
      ifs1 >> ur[i][j];
      ifs2 >> ui[i][j];
      /*
      opt.ur0[i][j] = opt.ur0[i][j] *
      10000000;//因为要保留最后2位数，且是对最后一位进行四舍五入就是操作最后3位数字；就放大1000
            t =
      (opt.ur0[i][j]+5)/10;//最后一位数+5，再取整；把多余的小数位通过取整舍去；
            opt.ur0[i][j] = (double)t/1000000; //别忘了把t的类型转化一下；
      opt.ui0[i][j] = opt.ui0[i][j] *
      10000000;//因为要保留最后2位数，且是对最后一位进行四舍五入就是操作最后3位数字；就放大1000
            t =
      (opt.ui0[i][j]+5)/10;//最后一位数+5，再取整；把多余的小数位通过取整舍去；
            opt.ui0[i][j] = (double)t/1000000; //别忘了把t的类型转化一下；
      */
    }
  }

  for (int i = 0; i < INPUT.n_grid; i++) {
    for (int j = 0; j < INPUT.n_grid; j++) {
      EXPECT_NEAR(ur[i][j], opt.ur[i][j], 1e5);
      EXPECT_NEAR(ui[i][j], opt.ui[i][j], 1e5);
      /*
      EXPECT_DOUBLE_EQ(ur[i][j], opt.ur0[i][j]);
      EXPECT_DOUBLE_EQ(ui[i][j], opt.ui0[i][j]);
      */
    }
  }
}
TEST(optical_field, Init_Phase_type_random) {

  EXPECT_TRUE(OPT::Init_Phase(INPUT, opt,0.2391,"random"));
}
/*
TEST(optical_field, Init_Phase_type_false) {

  EXPECT_FALSE(OPT::Init_Phase(INPUT, opt,0.2391,"rand"));
}
*/




} // namespace