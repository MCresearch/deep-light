#include "../gaussian.h"
#include "../input.h"
#include "../spot.h"
#include <gtest/gtest.h>

TEST(Gaussian_test, gaussian_fit_evaluate)
{
    //INPUT.read();

    //testing::InitGoogleTest(&argc, argv);

    //cout << INPUT.width << " " << INPUT.window_width  << endl;

    ifstream ifs_gaussian("single_gaussian3.txt");

    double** spot3 = new double*[3];
    double** spot3_fit = new double*[3];
    for (int iy=0; iy<3; iy++)
    {
        spot3[iy] = new double[3];
        spot3_fit[iy] = new double[3];
    }
    for (int iy=0; iy<3; iy++)
    {
        for (int ix=0; ix<3; ix++)
        {
            ifs_gaussian >> spot3[iy][ix];
        }
    }
    
    Gaussian gaussian;
    gaussian.fit(spot3, 3);
    gaussian.gaussian_window(spot3_fit, 3);
    for (int iy=0; iy<3; iy++)
    {
        for (int ix=0; ix<3; ix++)
        {
            EXPECT_FLOAT_EQ(spot3_fit[iy][ix], spot3[iy][ix]);
        }
        cout << endl;
    }

    for (int iy=0; iy<3; iy++)
    {
        delete[] spot3[iy];
        delete[] spot3_fit[iy];
    }
    delete[] spot3;
    delete[] spot3_fit;

    //return 0;
}

TEST(Spot_test, spot_gaussian_fit)
{
    INPUT.read();
    Spot spot;
    ifstream ifs;
    ifs.open("outIntensity.dat");
    ifstream ifs_predict("fitted_intensity.txt");
    //cout << INPUT.intensity_thre << endl;
    spot.readin(ifs);
    spot.identify_local_max();
    spot.fit_gaussian();
    spot.predict();
    //cout << 1 << endl;
    for (int iy=0; iy<INPUT.width; iy++)
    {
        for (int ix=0; ix<INPUT.width; ix++)
        {
            double intens;
            ifs_predict >> intens;
            EXPECT_FLOAT_EQ(spot.fitted_value[iy][ix], intens);
        }
    }
    spot.clean();
    ifs.close();
    ifs_predict.close();
}

int main(int argc, char ** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}