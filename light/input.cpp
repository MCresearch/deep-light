#include "input.h"

// Input INPUT;

Input::Input()  //
{
    mm = -1;
    n_grid = -1;
    n9 = -1;
    n1 = -1;

    mgs = -1;

    a0 = 0.0;
    xx0 = 0.0;
    aa0 = 0.0;
    dxy0 = 0.0;

    plm = 0.0;
    zfh = 0.0;

    airy = 0.0;
    xxz = 0.0;
    aaz = 0.0;
    dxyz = 0.0;

    minZnkDim = -1;
    maxZnkOrder = -1;

    rms = 0.0;
    eeznk = 0.0;
    Phase_option = "";
    dir = "";

    num_datas = 0;

    out_inIntensity = -1;
    out_zernike_coeff = -1;
    out_inPhase = -1;
    out_focusing = -1;
    out_mdfph1 = -1;
    out_my_fft2d1 = -1;
    out_evol1 = -1;
    out_my_fft2d2 = -1;
    out_mdfph2 = -1;
    out_outIntensity = -1;
}

Input::~Input() {}

bool Input::INIT(Input &INPUT)  //
{
    ifstream ifs("INPUT.txt");
    string   word1;
    string   word2;
    if (!ifs)
    {
        cout << "Error in reading INPUT file !" << endl;
        exit(0);
    }
    while (ifs >> word1)
    {
        char word3[40];
        strcpy(word3, word1.c_str());
        if (strcmp(word3, "mm") == 0)
        {
            ifs >> INPUT.mm;
            ifs.ignore(150, '\n');
        }

        else if (strcmp(word3, "mgs") == 0)
        {
            ifs >> INPUT.mgs;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "a0") == 0)
        {
            ifs >> INPUT.a0;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "xx0") == 0)
        {
            ifs >> INPUT.xx0;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "plm") == 0)
        {
            ifs >> INPUT.plm;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "zfh") == 0)
        {
            ifs >> INPUT.zfh;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "xxz") == 0)
        {
            ifs >> INPUT.xxz;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "minZnkDim") == 0)
        {
            ifs >> INPUT.minZnkDim;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "maxZnkOrder") == 0)
        {
            ifs >> INPUT.maxZnkOrder;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "rms") == 0)
        {
            ifs >> INPUT.rms;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "eeznk") == 0)
        {
            ifs >> INPUT.eeznk;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "Phase_option") == 0)
        {
            ifs >> INPUT.Phase_option;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "dir") == 0)
        {
            ifs >> INPUT.dir;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "num_datas") == 0)
        {
            ifs >> INPUT.num_datas;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_inIntensity") == 0)
        {
            ifs >> INPUT.out_inIntensity;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_zernike_coeff") == 0)
        {
            ifs >> INPUT.out_zernike_coeff;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_inPhase") == 0)
        {
            ifs >> INPUT.out_inPhase;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_focusing") == 0)
        {
            ifs >> INPUT.out_focusing;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_mdfph1") == 0)
        {
            ifs >> INPUT.out_mdfph1;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_my_fft2d1") == 0)
        {
            ifs >> INPUT.out_my_fft2d1;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_evol1") == 0)
        {
            ifs >> INPUT.out_evol1;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_my_fft2d2") == 0)
        {
            ifs >> INPUT.out_my_fft2d2;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_mdfph2") == 0)
        {
            ifs >> INPUT.out_mdfph2;
            ifs.ignore(150, '\n');
        }
        else if (strcmp(word3, "out_outIntensity") == 0)
        {
            ifs >> INPUT.out_outIntensity;
            ifs.ignore(150, '\n');
        }
        else
        {
            cout << "the paremeter name:" << word1 << "is not used!" << endl;
            ifs.ignore(150, '\n');
        }

        INPUT.n_grid = pow(2, INPUT.mm);
        INPUT.n9 = INPUT.n_grid + 9;
        INPUT.n1 = INPUT.n_grid / 2 + 1;

        INPUT.aa0 = INPUT.xx0 * INPUT.a0;
        INPUT.dxy0 = INPUT.aa0 / INPUT.n_grid;

        INPUT.airy = 1.22 * INPUT.plm * INPUT.zfh / (2 * INPUT.a0);

        INPUT.aaz = INPUT.airy * INPUT.xxz;
        INPUT.dxyz = INPUT.aaz / INPUT.n_grid;
    }

    if (INPUT.mm == -1)
    {
        cout << "please input mm." << endl;
        return false;
    }
    if (INPUT.n_grid == -1)
    {
        cout << "please input n_grmm." << endl;
        return false;
    }
    if (INPUT.n9 == -1)
    {
        cout << "please input n9." << endl;
        return false;
    }
    if (INPUT.n1 == -1)
    {
        cout << "please input n1." << endl;
        return false;
    }
    if (INPUT.mgs == -1)
    {
        cout << "please input mgs." << endl;
        return false;
    }
    if (INPUT.a0 == 0.0)
    {
        cout << "please input a0." << endl;
        return false;
    }
    if (INPUT.xx0 == 0.0)
    {
        cout << "please input xx0." << endl;
        return false;
    }
    if (INPUT.aa0 == 0.0)
    {
        cout << "please input aa0." << endl;
        return false;
    }
    if (INPUT.dxy0 == 0.0)
    {
        cout << "please input dxy0." << endl;
        return false;
    }
    if (INPUT.plm == 0.0)
    {
        cout << "please input plm." << endl;
        return false;
    }
    if (INPUT.zfh == 0.0)
    {
        cout << "please input zfh." << endl;
        return false;
    }
    if (INPUT.airy == 0.0)
    {
        cout << "please input airy." << endl;
        return false;
    }
    if (INPUT.xxz == 0.0)
    {
        cout << "please input xxz." << endl;
        return false;
    }
    if (INPUT.aaz == 0.0)
    {
        cout << "please input aaz." << endl;
        return false;
    }
    if (INPUT.dxyz == 0.0)
    {
        cout << "please input dxy." << endl;
        return false;
    }
    if (INPUT.minZnkDim == -1)
    {
        cout << "please input minZnkDim." << endl;
        return false;
    }
    if (INPUT.maxZnkOrder == -1)
    {
        cout << "please input maxZnkOrder." << endl;
        return false;
    }
    if (INPUT.rms == 0.0)
    {
        cout << "please input rms." << endl;
        return false;
    }
    if (INPUT.eeznk == 0.0)
    {
        cout << "please input eeznk." << endl;
        return false;
    }
    if (INPUT.Phase_option.empty())
    {
        cout << "please input Phase_option." << endl;
        return false;
    }
    if (INPUT.dir.empty())
    {
        cout << "please input dir." << endl;
        return false;
    }
    if (INPUT.num_datas == 0)
    {
        cout << "please input num_datas." << endl;
        return false;
    }
    if (INPUT.out_inIntensity == -1)
    {
        cout << "please input out_inIntensity." << endl;
        return false;
    }
    if (INPUT.out_zernike_coeff == -1)
    {
        cout << "please input out_zernike_coeff ." << endl;
        return false;
    }
    if (INPUT.out_inPhase == -1)
    {
        cout << "please input out_inPhase  ." << endl;
        return false;
    }
    if (INPUT.out_focusing == -1)
    {
        cout << "please input out_focusing  ." << endl;
        return false;
    }
    if (INPUT.out_mdfph1 == -1)
    {
        cout << "please input out_mdfph1  ." << endl;
        return false;
    }
    if (INPUT.out_my_fft2d1 == -1)
    {
        cout << "please input out_my_fft2d1  ." << endl;
        return false;
    }
    if (INPUT.out_evol1 == -1)
    {
        cout << "please input out_evol1   ." << endl;
        return false;
    }
    if (INPUT.out_my_fft2d2 == -1)
    {
        cout << "please input out_my_fft2d2   ." << endl;
        return false;
    }
    if (INPUT.out_mdfph2 == -1)
    {
        cout << "please input out_mdfph2    ." << endl;
        return false;
    }
    if (INPUT.out_outIntensity == -1)
    {
        cout << "please input out_outIntensity    ." << endl;
        return false;
    }
    return true;
}
