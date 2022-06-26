#include <iostream>
#include <fstream>
#include <string.h>
#include <cstdlib>
#include <iomanip>
#include <stdio.h>

using namespace std;

#ifndef INPUT_H
#define INPUT_H

class Input
{
    public:
    
    Input();
    ~Input();


    void read();

    int width;
    int window_width;
    double intensity_thre;
    int write_local_max;
    int write_predicted_intensity;



    template <class T>
    static void read_value(ifstream &ifs, T &var)
    {
        ifs >> var;
        ifs.ignore(150, '\n');
        return;
    }

};

extern Input INPUT;
extern ofstream ofs_running;

#endif