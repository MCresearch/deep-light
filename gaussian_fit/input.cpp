#include "input.h"

Input INPUT;
ofstream ofs_running;

Input::Input()
{
    width = -1;
    window_width = -1;
    intensity_thre = 0.0;
    write_local_max = 0;
    write_predicted_intensity = 0;
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
    }
    return;
}