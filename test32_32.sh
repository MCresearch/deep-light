#!/bin/bash
cd tests
cd 32_32/
rm -f dl*
rm -f *.exe
cd /home/xianyuer/yuer/num/tests/fft/
rm -f dl*
cd /home/xianyuer/yuer/num/
make
cp ./light.exe /home/xianyuer/yuer/num/tests/32_32/
cd /home/xianyuer/yuer/num/tests/32_32/
./light.exe

cd /home/xianyuer/yuer/num/

make clean
