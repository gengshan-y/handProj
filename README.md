# Multi-person Pose Tracking
This is the batch version of multi-person pose tracking C++ code.

We used *faster-rcnn* to detect objects. We used *HOG+SVM* to obtain appearance similarity and *Kalman filter* to model motion dynamics. Summing up the scores from appearance model and motion model, we associate detections to tracking objects.  Then [*Convulutional Pose Machines*](https://github.com/shihenw/convolutional-pose-machines-release) was used to estimate joint (currently wrist) postions.

## Features
- concurrency by pthreads mutex
- batch implementation of convolutional pose machine 

## Usage
- `make`
- clear output data folder `rm -rf data/*`
- run `./main  config.c`
- run genVideo.ipynb to generate example videos

### Pre-requisites
- opencv 2.4.x
- caffe-faster-rcnn
- libconfig
- faster-rcnn model/weights && convolutional pose machines model/weights

### Run-time libs
- export PYTHONPATH=/home/gengshan/workDec/threadProc/lib:/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/python
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/build/lib
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gengshan/workOct/cudnn_v3/lib64
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/softwares/libconfig/lib/

### Compile-time libs
- export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/cuda-7.0/pkgconfig

## Notes
- appearance model may be substituted by Siamese network for better tracking performance
- tune convolution pose machine input scale
- should use `ffmpeg -r 5 -i %04d.jpg -vb 20M frame.mpg` to get better image quality

## Acknowledgement
- faster-rcnn c++ implementation is based on [this blog](http://blog.csdn.net/xyy19920105/article/details/50440957)
- parameter configuration part is based on [this code](https://github.com/gnebehay/HoughTrack)
