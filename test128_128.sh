#!/bin/bash
cd tests
cd 128_128/
rm -f dl*
rm -f *.exe
cd /home/xianyuer/yuer/num/tests/fft/
rm -f dl*
cd /home/xianyuer/yuer/num/
make
cp ./light.exe /home/xianyuer/yuer/num/tests/128_128/
cd /home/xianyuer/yuer/num/tests/128_128/
./light.exe

cd /home/xianyuer/yuer/num/

make clean