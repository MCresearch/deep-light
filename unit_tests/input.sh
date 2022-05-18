#!/bin/bash
g++ input_unittest.cc ../light/input.cpp -o input ../light/input.h -lgtest -lgtest_main -lpthread
./input  