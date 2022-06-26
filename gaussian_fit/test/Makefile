VPATH=./\
../\

OBJS = test_gaussian_fit.o\
../input.o\
../gaussian.o\
../spot.o\

CC=mpiicc

.cpp.o:
		${CC} ${OPTION} -c -O3 $< -o $@ -I/usr/src/gtest/include/gtest -L/usr/src/gtest -lgtest -lpthread 

gaussian_test :${OBJS}
		${CC} ${OPTION} -o gaussian_test.exe ${OBJS} -I/usr/src/gtest/include/gtest -L/usr/src/gtest -lgtest -lpthread

clean:
		rm -f *.o *.exe