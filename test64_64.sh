#!/bin/bash
cd tests
cd 8_8/
rm -f dl*
rm -f *.exe
cd /home/xianyuer/yuer/num/
make
cp ./light.exe /home/xianyuer/yuer/num/tests/8_8/
cd /home/xianyuer/yuer/num/tests/8_8/
./light.exe
cd /home/xianyuer/yuer/num/
make clean
