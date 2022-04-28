//==================================================
// Main function：The far-field transmission of focused beam is realized by fast
// Fourier transform and coordinate adaptation transform. Date: 2022-02-14
//==================================================
#include "FFt.h"
#include "Zernike.h"
#include "fun.h"
#include "input.h"
#include "optical_field.h"
#include <time.h>
#include <random>

int main()
{
    cout << "111" << endl;
    Input INPUT;
    if (!INPUT.INIT(INPUT))
    {
        cout << "input error!" << endl;
        exit(0);
    }

    OPT opt;
    OPT::Init_Intensity(INPUT, opt);

    double a1 = 0.0;
    default_random_engine            random(a1);
    std::normal_distribution<double> dis(0, 1);
    for (int i = 0; i < INPUT.num_datas; i++)
    {
        
        a1 = dis(random);
        OPT::Init_Phase(INPUT, opt, a1, INPUT.Phase_option);  // add for "random" or "confirm"

        OPT::numercial_diffraction(INPUT, a1, opt);
    }


    // start time
    cout << "The current time is: " << (double)clock() << "s" << endl;
    // time
    cout << "The run time is: " << (double)clock() / CLOCKS_PER_SEC << "s" << endl;
    return 0;
}
