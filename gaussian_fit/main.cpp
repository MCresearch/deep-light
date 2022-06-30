#include "gaussian.h"
#include "input.h"
#include "spot.h"

int main(int argc, char **argv)
{
    INPUT.read();
    ofs_running.open("running.log");
    ofs_local_max.open("local_max.txt");
    ofs_predict.open("predicted_intensity.txt");

    Spot spot;
    ifstream ifs;
    ifs.open(INPUT.intensity_file);
    //cout << INPUT.intensity_thre << endl;
    for (int isnapshot=0; isnapshot<INPUT.nsnapshot; isnapshot++)
    {
        spot.readin(ifs);
        if (INPUT.write_local_max > 0)
        {
            ofs_running << "isnapshot=" << isnapshot << endl;
            cout << "isnapshot=" << isnapshot << endl;
        }
        if (INPUT.top_pctg > 0)
        {
            spot.calc_quantile();
        }
        else
        {
            spot.quantile = INPUT.intensity_thre;
        }
        ofs_running << "intensity_thre=" << spot.quantile << endl;
        spot.identify_local_max();
        if (INPUT.write_local_max)
        {
            ofs_local_max << "isnapshot=" << isnapshot << " " << spot.nlocal_max << endl;
        }
        spot.fit_gaussian();
        spot.predict();
        spot.clean();
    }
    ofs_running.close();
    ofs_local_max.close();
    ofs_predict.close();

    return 0;
}