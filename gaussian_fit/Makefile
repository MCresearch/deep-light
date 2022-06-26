OBJS = main.o\
input.o\
gaussian.o\
spot.o\

CC=mpiicc

.cpp.o:
		${CC} ${OPTION} -c -O3 $< -o $@

spot_fit :${OBJS}
		${CC} ${OPTION} -o spot_fit.exe ${OBJS}

clean:
		rm -f *.o *.exe