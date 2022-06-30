#include "input.h"

Input INPUT;
ofstream ofs_running;
ofstream ofs_local_max;
ofstream ofs_predict;

Input::Input()
{
    width = -1;
    window_width = -1;
    intensity_thre = 0.0;
    write_local_max = 0;
    write_predicted_intensity = 0;
    top_pctg = 0.0;
}

Input::~Input(){}

void Input::read()
{
    ifstream ifs("INPUT");
    char keyword[80];
    while(ifs.good())
    {
        ifs >> keyword;
        if (strcmp(keyword, "width") == 0) read_value(ifs, width);
        else if (strcmp(keyword, "window_width") == 0) read_value(ifs, window_width);
        else if (strcmp(keyword, "intensity_thre") == 0) read_value(ifs, intensity_thre);
        else if (strcmp(keyword, "write_local_max") == 0) read_value(ifs, write_local_max);
        else if (strcmp(keyword, "write_predicted_intensity") == 0) read_value(ifs, write_predicted_intensity);
        else if (strcmp(keyword, "intensity_file") == 0) read_value(ifs, intensity_file);
        else if (strcmp(keyword, "nsnapshot") == 0) read_value(ifs, nsnapshot);
        else if (strcmp(keyword, "top_pctg") == 0) read_value(ifs, top_pctg);

    }
    return;
}