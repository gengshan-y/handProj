## run-time libs
export PYTHONPATH=/home/gengshan/workDec/threadProc/lib:/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/build/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gengshan/workOct/cudnn_v3/lib64

## compile-time libs
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/cuda-7.0/pkgconfig

 ./headTracking rtsp://admin:a123456@192.168.61.102 y


faster-rcnn c++ implementation is base on [this blog](http://blog.csdn.net/xyy19920105/article/details/50440957)

## Issues
- detection and tarcking only works well under constrained scenarios
