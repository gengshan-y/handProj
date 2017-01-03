## Multi-Hands Tracking by multiple object tracking and pose estimation
This is the batch version of multi-hands tracking C++ code. We used *faster-rcnn* to detect objects, *HOG+SVM* to model appearance features and *Kalman filter* to model motion dynamics. *(Convulutional Pose Machines)*[https://github.com/shihenw/convolutional-pose-machines-release] was used to estimate wrist postions.

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

## Usage
- clear output data folder `rm -rf data/*`
- run `./main  config.c`
- run genVideo.ipynb to generate example videos 

## TODO
- appearance model may be substituted by Siamese network for better performance  
- should use `ffmpeg -r 5 -i %04d.jpg -vb 20M frame.mpg` to get better image quality

## Acknowledgement
- faster-rcnn c++ implementation is based on (this blog)[http://blog.csdn.net/xyy19920105/article/details/50440957]
- parameter configuration part is based on (this code)[https://github.com/gnebehay/HoughTrack]
