#include "gaussian.h"
#include "input.h"
#include "spot.h"
#include <gtest/gtest.h>

TEST(Quantile_test, quantile)
{
    cout << 0 << endl;
    INPUT.read();
    Spot spot;
    ifstream ifs;
    cout << 1 << endl;
    ifs.open(INPUT.intensity_file);
    spot.readin(ifs);
    cout << 2 << endl;
    ifs.close();
    spot.calc_quantile();
    EXPECT_FLOAT_EQ(spot.quantile, 9.86382);
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}