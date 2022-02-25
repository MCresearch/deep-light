#!/bin/bash
rm -r bulid
rm -r -f ./result.txt


cmake -B bulid
cd bulid
make
cp -f  mylight ../
cd ..

