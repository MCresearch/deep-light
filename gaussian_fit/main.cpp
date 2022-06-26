#include "gaussian.h"
#include "input.h"
#include "spot.h"

int main(int argc, char **argv)
{
    INPUT.read();
    ofs_running.open("running.log");

    Spot spot;
    ifstream ifs;
    ifs.open("outIntensity.dat");
    cout << INPUT.intensity_thre << endl;
    spot.readin(ifs);
    spot.identify_local_max();
    spot.fit_gaussian();
    spot.predict();
    spot.clean();
    ofs_running.close();

    return 0;
}