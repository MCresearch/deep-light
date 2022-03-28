#!/bin/bash
cd tests
rm -f dl*
cd ..
make
./light.exe
make clean
