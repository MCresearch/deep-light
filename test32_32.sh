#!/bin/bash
cd tests
cd 32_32/
rm -f dl*
rm -f *.exe
cd ../..
make
cp ./light.exe ./tests/32_32/
cd ./tests/32_32/
./light.exe

cd ../..

make clean
