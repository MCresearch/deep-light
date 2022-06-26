#include "spot.h"

Spot::Spot(){}

Spot::~Spot(){}

void Spot::identify_local_max()
{
    int l = int(window_width/2);
    nlocal_max_ub = pow(int(width/window_width), 2);
    local_max_coord = new int*[nlocal_max_ub];
    for (int imax=0; imax<nlocal_max_ub; imax++)
    {
        local_max_coord[imax] = new int[2];
        local_max_coord[imax][0] = -1;
        local_max_coord[imax][1] = -1;
    }
    nlocal_max = 0;

    for (int iy=window_width; iy<width-window_width; iy++)
    {
        for (int ix=window_width; ix<width-window_width; ix++)
        {
            if (value[iy][ix] < this->local_max_thre)
            {
                continue;
            }
            bool flag = true;
            for (int iy1=-l; iy1<l+1; iy1++)
            {
                if (!flag) break;
                
                for (int ix1=-l; ix1<l+1; ix1++)
                {
                    if (ix1 == 0 and iy1 == 0) continue;
                    if (value[iy][ix] < value[iy+iy1][ix+ix1]) 
                    {
                        flag = false;
                        break;
                    }
                }
            }
            if (flag)
            {
                local_max_coord[nlocal_max][0] = iy;
                local_max_coord[nlocal_max][1] = ix;
                nlocal_max++;
                if (INPUT.write_local_max > 0)
                {
                    ofs_running << "local_max " << nlocal_max-1 << ", ix=" << ix << ", iy=" << iy << endl;
                    for (int iy1=-l; iy1<l+1; iy1++)
                    {
                        for (int ix1=-l; ix1<l+1; ix1++)
                        {
                            ofs_running << value[iy+iy1][ix+ix1] << " ";
                        }
                        ofs_running << endl;
                    }
                }
            }
        }
    }
    identify_ = true;
    return;
}

void Spot::fit_gaussian()
{
    int l = int(window_width/2);
    this->gaussian = new Gaussian[nlocal_max];
    double** local_window = new double*[window_width];
    for (int iy=0; iy<window_width; iy++)
    {
        local_window[iy] = new double[window_width];
    }
    for (int imax=0; imax<nlocal_max; imax++)
    {
        for (int iy=-l; iy<l+1; iy++)
        {
            for (int ix=-l; ix<l+1; ix++)
            {
                local_window[iy+l][ix+l] = value[local_max_coord[imax][0]+iy][local_max_coord[imax][1]+ix];
            }
        }
        gaussian[imax].fit(local_window, window_width);
    }
    fit_ = true;
    for (int iy=0; iy<this->window_width; iy++)
    {
        delete[] local_window[iy];
    }
    delete[] local_window;
    return;
}

void Spot::predict()
{
    this->fitted_value = new double*[width];
    for (int iy=0; iy<width; iy++)
    {
        fitted_value[iy] = new double[width];
        for (int ix=0; ix<width; ix++)
        {
            fitted_value[iy][ix] = 0;
        }
    }
    this->predict_ = true;

    for (int imax=0; imax<nlocal_max; imax++)
    {
        if (gaussian[imax].sigmax2 < 0 or gaussian[imax].sigmay2 < 0) continue;
        for (int iy=0; iy<width; iy++)
        {
            for (int ix=0; ix<width; ix++)
            {
                fitted_value[iy][ix] += gaussian[imax].eval(ix-local_max_coord[imax][1], iy-local_max_coord[imax][0]);
            }
        }
    }
    if (INPUT.write_predicted_intensity > 0)
    {
        ofstream ofs("fitted_intensity.txt");
        ofs << setprecision(8);
        for (int iy=0; iy<width; iy++)
        {
            for (int ix=0; ix<width; ix++)
            {
                ofs << fitted_value[iy][ix] << " ";
            }
            ofs << endl;
        }
        ofs.close();
    }

    return;
}

void Spot::clean()
{
    if (read_)
    {
        for (int iy=0; iy<width; iy++)
        {
            delete[] value[iy];
        }
        delete[] value;
    }
    if (identify_)
    {
        for (int imax=0; imax<nlocal_max_ub; imax++)
        {
            delete[] local_max_coord[imax];
        }
        delete[] local_max_coord;
    }
    if (fit_)
    {
        delete[] gaussian;
    }
    if (predict_)
    {
        for (int iy=0; iy<this->width; iy++)
        {
            delete[] fitted_value[iy];
        }
        delete[] fitted_value;
    }
    return;
}

void Spot::readin(ifstream &ifs)
{
    assert(INPUT.width > 0);
    assert(INPUT.window_width > 0);
    assert(INPUT.intensity_thre > 0);
    this->width = INPUT.width;
    this->window_width = INPUT.window_width;
    this->local_max_thre = INPUT.intensity_thre;
    this->value = new double*[width];
    for (int iy=0; iy<width; iy++) this->value[iy] = new double[width];
    for (int iy=0; iy<width; iy++)
    {
        for (int ix=0; ix<width; ix++)
        {
            ifs >> this->value[iy][ix];
        }
    }
    read_ = true;
    return;
}