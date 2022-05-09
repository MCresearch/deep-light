VPATH = ./light
TARGET:=light.exe

OBJS=main.o\


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



