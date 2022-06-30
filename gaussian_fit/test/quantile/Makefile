VPATH=./\
../../\

OBJS = test_quantile.o\
../../input.o\
../../gaussian.o\
../../spot.o\

CC=mpiicc

.cpp.o:
		${CC} ${OPTION} -c -O3 $< -o $@ -I../../ -I/usr/src/gtest/include/gtest -L/usr/src/gtest -lgtest -lpthread 

quantile_test:${OBJS}
		${CC} ${OPTION} -o quantile_test.exe ${OBJS} -I/usr/src/gtest/include/gtest -L/usr/src/gtest -lgtest -lpthread

clean:
		rm -f *.o *.exe