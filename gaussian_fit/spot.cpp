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
            if (value[iy][ix] < this->quantile)
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
                    ofs_running << nlocal_max-1 << " " << ix << " " << iy << endl;
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
        if (INPUT.write_local_max > 0)
        {
            ofs_local_max << imax << " " << local_max_coord[imax][1] << " " << local_max_coord[imax][0] << " " << gaussian[imax].I << " "
             << gaussian[imax].x0 << " " << gaussian[imax].y0 << " " << gaussian[imax].sigmax2 << " " << gaussian[imax].sigmay2 << " " << gaussian[imax].alpha << endl;
        }
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
        if (abs(gaussian[imax].x0) > 2 or abs(gaussian[imax].y0) > 2) continue;
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
        //ofstream ofs("fitted_intensity.txt");
        ofs_predict << setprecision(8);
        for (int iy=0; iy<width; iy++)
        {
            for (int ix=0; ix<width; ix++)
            {
                ofs_predict << fitted_value[iy][ix] << " ";
            }
            ofs_predict << endl;
        }
        //ofs.close();
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
    assert(INPUT.intensity_thre > 0 or INPUT.top_pctg > 0);
    this->width = INPUT.width;
    this->window_width = INPUT.window_width;
    //this->local_max_thre = INPUT.intensity_thre;
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

void Spot::calc_quantile()
{
    // This is an incomplete bubble sorting, since we only have to
    // sort the first INPUT.top_pctg*width*width times to get the 
    // quantile.
    assert(this->read_);
    int length = this->width*this->width;
    double* intensity1D = new double[length];
    for (int iy=0; iy<this->width; iy++)
    {
        for (int ix=0; ix<this->width; ix++)
        {
            intensity1D[iy*this->width + ix] = this->value[iy][ix];
        }
    }

    int Ntop_pctg = int(INPUT.top_pctg * length)+1;
    //cout << "Ntop_pctg=" << Ntop_pctg << endl;
    for (int ii=0; ii<Ntop_pctg; ii++)
    {
        //cout << "ii=" << ii << endl;
        for (int jj=length-1; jj>ii; jj--)
        {
            if (intensity1D[jj] > intensity1D[jj-1])
            {
                double tmp=intensity1D[jj-1];
                intensity1D[jj-1] = intensity1D[jj];
                intensity1D[jj] = tmp;
            }
        }
    }
    //cout << "choosed order = " << Ntop_pctg << endl;
    this->quantile = intensity1D[Ntop_pctg-1];
    INPUT.intensity_thre = this->quantile;
    //cout << "intensity_thre=" << this->quantile << endl;

    delete[] intensity1D;
    return;
}