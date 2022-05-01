##############################################################################
# MAKEFILE
##############################################################################
#INCLUDESLIB = -lfftw3
##INCLUDESH = -I/usr/local/fftw/include
#INCLUDESH=-I/home/xianyuer/soft/fftw-3.3.10/yuer/include/
##LIBS = -L/usr/local/fftw/lib
#LIBS=-L/home/xianyuer/soft/fftw-3.3.10/yuer/lib/
VPATH = ./light
TARGET:=light.exe

OBJS=main.o\
input.o\
fun.o\
FFt.o\
Zernike.o\
optical_field.o\

#${TARGET}:${OBJS}
#		${CXX} $^ -o $@  
#		${CXX} $^ -o $@ ${INCLUDESH} ${LIBS} ${INCLUDESLIB}  
#clean:
#	rm -f *.o *.exe


${TARGET}:${OBJS}
#               ${CXX} $^ -o $@
		${CXX} $^ ${INCLUDESH} ${LIBS} ${INCLUDESLIB} -O3 -o $@
%.o:%.cpp
		${CXX} $^ ${INCLUDESH} ${LIBS} ${INCLUDESLIB} -O3 -c
clean:
		rm -f *.o *.exe



