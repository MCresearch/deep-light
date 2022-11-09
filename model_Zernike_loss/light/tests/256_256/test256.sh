#!/bin/bash
rm -f dl*
rm -f *.exe
cd ../..
make
cp ./light.exe ./tests/256_256/
cd ./tests/256_256/
./light.exe

cd ../..

make clean
