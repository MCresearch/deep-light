#include "../light/input.h"

#include <limits.h>

#include "gtest/gtest.h"
namespace {

Input INPUT;

TEST(INPUT, Input0) {

  EXPECT_EQ(-1, INPUT.mm);
  EXPECT_EQ(-1, INPUT.n_grid);
  EXPECT_EQ(-1, INPUT.n9);
  EXPECT_EQ(-1, INPUT.n1);
  EXPECT_EQ(-1, INPUT.mgs);
  EXPECT_EQ(0, INPUT.a0);
  EXPECT_EQ(0, INPUT.xx0);
  EXPECT_EQ(0, INPUT.aa0);
  EXPECT_EQ(0, INPUT.dxy0);
  EXPECT_EQ(0, INPUT.plm);
  EXPECT_EQ(0, INPUT.zfh);
  EXPECT_EQ(0, INPUT.airy);
  EXPECT_EQ(0, INPUT.xxz);
  EXPECT_EQ(0, INPUT.aaz);
  EXPECT_EQ(0, INPUT.dxyz);
  EXPECT_EQ(-1, INPUT.minZnkDim);
  EXPECT_EQ(-1, INPUT.maxZnkOrder);
  EXPECT_EQ(0, INPUT.rms);
  EXPECT_EQ(0, INPUT.eeznk);
  EXPECT_TRUE(INPUT.Phase_option.empty());
  EXPECT_TRUE(INPUT.dir.empty());
  EXPECT_EQ(1, INPUT.num_datas);
  EXPECT_EQ(-1, INPUT.out_inIntensity);
  EXPECT_EQ(-1, INPUT.out_zernike_coeff);
  EXPECT_EQ(-1, INPUT.out_inPhase);
  EXPECT_EQ(-1, INPUT.out_focusing);
  EXPECT_EQ(-1, INPUT.out_mdfph1);
  EXPECT_EQ(-1, INPUT.out_my_fft2d1);
  EXPECT_EQ(-1, INPUT.out_evol1);
  EXPECT_EQ(-1, INPUT.out_my_fft2d2);
  EXPECT_EQ(-1, INPUT.out_mdfph2);
  EXPECT_EQ(-1, INPUT.out_outIntensity);
}
TEST(INPUT, InputINIT) {
  // INPUT.INIT(INPUT);
  EXPECT_TRUE(INPUT.INIT(INPUT));
  EXPECT_EQ(5, INPUT.mm);
  EXPECT_EQ(32, INPUT.n_grid);
  EXPECT_EQ(32 + 9, INPUT.n9);
  EXPECT_EQ(32 / 2 + 1, INPUT.n1);
  EXPECT_EQ(8, INPUT.mgs);
  EXPECT_EQ(0.3, INPUT.a0);
  EXPECT_EQ(4, INPUT.xx0);
  EXPECT_EQ(4 * 0.3, INPUT.aa0);
  EXPECT_EQ(4 * 0.3 / 32, INPUT.dxy0);
  EXPECT_EQ(1e-6, INPUT.plm);
  EXPECT_EQ(3e3, INPUT.zfh);
  EXPECT_EQ(1.22 * 1e-6 * 3e3 / (2 * 0.3), INPUT.airy);
  EXPECT_EQ(20, INPUT.xxz);
  EXPECT_FLOAT_EQ(20 * 1.22 * 1e-6 * 3e3 / (2 * 0.3), INPUT.aaz);
  EXPECT_FLOAT_EQ(20 * 1.22 * 1e-6 * 3e3 / (2 * 0.3) / 32, INPUT.dxyz);
  EXPECT_EQ(1, INPUT.minZnkDim);
  EXPECT_EQ(3, INPUT.maxZnkOrder);
  EXPECT_EQ(1, INPUT.rms);
  EXPECT_EQ(0.2, INPUT.eeznk);
  string a1 = "random";
  string a2 = "./";
  EXPECT_EQ(0, a1.compare(INPUT.Phase_option));
  EXPECT_EQ(0, a2.compare(INPUT.dir));
  EXPECT_EQ(100, INPUT.num_datas);
  EXPECT_EQ(1, INPUT.out_inIntensity);
  EXPECT_EQ(1, INPUT.out_zernike_coeff);
  EXPECT_EQ(0, INPUT.out_inPhase);
  EXPECT_EQ(0, INPUT.out_focusing);
  EXPECT_EQ(0, INPUT.out_mdfph1);
  EXPECT_EQ(0, INPUT.out_my_fft2d1);
  EXPECT_EQ(0, INPUT.out_evol1);
  EXPECT_EQ(0, INPUT.out_my_fft2d2);
  EXPECT_EQ(0, INPUT.out_mdfph2);
  EXPECT_EQ(1, INPUT.out_outIntensity);
}
} // namespace
