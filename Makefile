##############################################################################
# MAKEFILE
##############################################################################
#INCLUDESLIB = -lfftw3
##INCLUDESH = -I/usr/local/fftw/include
#INCLUDESH=-I/home/xianyuer/soft/fftw-3.3.10/yuer/include/
##LIBS = -L/usr/local/fftw/lib
#LIBS=-L/home/xianyuer/soft/fftw-3.3.10/yuer/lib/
TARGET:=light.exe

VPATH = light

OBJS=main.o\
input.o\
fun.o\
FFt.o\
Zernike.o\
optical_field.o\

${TARGET}:${OBJS}
		${CXX} $^ -o $@ 
#		${CXX} $^ -o $@ ${INCLUDESH} ${LIBS} ${INCLUDESLIB}  
clean:
	rm -f *.o *.exe




