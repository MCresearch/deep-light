#!/bin/bash
rm -f dl*
rm -f *.exe
cd ../..
make
cp ./light.exe ./tests/64_64/
cd ./tests/64_64/
./light.exe
cd ../..
make clean
