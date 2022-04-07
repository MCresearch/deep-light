#!/bin/bash
cd tests
cd 256_256/
rm -f dl*
rm -f *.exe
cd /home/xianyuer/yuer/num/tests/fft/
rm -f dl*
cd /home/xianyuer/yuer/num/
make
cp ./light.exe /home/xianyuer/yuer/num/tests/256_256/
cd /home/xianyuer/yuer/num/tests/256_256/
./light.exe

cd /home/xianyuer/yuer/num/

make clean