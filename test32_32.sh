#!/bin/bash
cd tests
cd 32_32/
rm -f dl*
rm -f *.exe
cd /home/xianyuer/yuer/num_mechinelearning/
make
cp ./light.exe /home/xianyuer/yuer/num_mechinelearning/tests/32_32/
cd /home/xianyuer/yuer/num_mechinelearning/tests/32_32/
./light.exe

cd /home/xianyuer/yuer/num_mechinelearning/

make clean
