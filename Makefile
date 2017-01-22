CC=g++
CXXFLAGS=-g -std=c++0x -Wall `pkg-config opencv --cflags`  
# -gdwarf-3 for different format of debugging info
# -Wall to enable all compiler's warning messages
# c++0x partially support c++11 features

LDFLAGS=`pkg-config opencv --libs`  # -g for debugging symbols

CXXFLAGS_FA=-g -std=c++0x -Wall `pkg-config opencv cublas-7.0 cuda-7.0 cudart-7.0 --cflags`
CXXFLAGS_FA:=${CXXFLAGS_FA} -I/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/include -I/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/.build_release/src
LDFLAGS_FA=`pkg-config cudart-7.0 cublas-7.0 cuda-7.0 opencv libglog --libs` 
LDFLAGS_FA:=${LDFLAGS_FA} -L/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/build/lib -lcaffe -L/home/gengshan/workOct/cudnn_v3/lib64 -lcudnn

all: clean main

main: global.o cvLib.o Tracker.o imgSVM.o cmpLib.o nms.o caffeDet.o draw.o
	${CC} -o main *.o main.cpp ${LDFLAGS} ${CXXFLAGS}  ${CXXFLAGS_FA}  ${LDFLAGS_FA}
#  use *.o to link with all .o files, otherwise will ignore all .o files

global.o: global.hpp

cvLib.o: cvLib.hpp

cmpLib.o: cmpLib.hpp

Tracker.o: Tracker.hpp

imgSVM.o: imgSVM.hpp

nms.o: 
	nvcc -c nms_kernel.cu -o nms.o

caffeDet.o:
	g++ -c caffeDet.cpp -o caffeDet.o ${CXXFLAGS_FA}

draw.o: draw.hpp

# for test
test: clean draw.o global.o cvLib.o Tracker.o imgSVM.o
	${CC} -o test *.o test.cpp ${LDFLAGS} ${CXXFLAGS}

clean: 
	rm -f main *.o
