**Under development**

## Pre-requisites
caffe-faster-rcnn
libconfig

## Run-time libs
export PYTHONPATH=/home/gengshan/workDec/threadProc/lib:/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/build/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gengshan/workOct/cudnn_v3/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/softwares/libconfig/lib/

## Compile-time libs
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/cuda-7.0/pkgconfig

## Notes

faster-rcnn c++ implementation is base on [this blog](http://blog.csdn.net/xyy19920105/article/details/50440957)

## Issues
- detection and tarcking only works well under constrained scenarios
- should use `ffmpeg -r 5 -i %04d.jpg -vb 20M frame.mpg` to get better image quality

## Usage
`rm -rf data/*;./main  config.c`
