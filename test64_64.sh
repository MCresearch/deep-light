#!/bin/bash
cd tests
cd 64_64/
rm -f dl*
rm -f *.exe
cd /home/xianyuer/yuer/num/
make
cp ./light.exe /home/xianyuer/yuer/num/tests/64_64/
cd /home/xianyuer/yuer/num/tests/64_64/
./light.exe
cd /home/xianyuer/yuer/num/
make clean
