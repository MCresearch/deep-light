#!/bin/bash
rm -f dl*
rm -f *.exe
cd ../..
make
cp ./light.exe ./tests/128_128/
cd ./tests/128_128/
./light.exe

cd ../..

make clean