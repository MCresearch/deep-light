#!/bin/bash
cd test
rm -f dl*
cd ..
make
./light.exe
make clean
