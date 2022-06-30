#!/bin/bash
rm -f dl*
rm -f *.exe
cd ../../
make
cp ./light.exe ./tests/test/
cd ./tests/test/
./light.exe
cd ../../
make clean
