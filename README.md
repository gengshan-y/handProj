**Under development**

## Multi-Hands Tracking by multiple object tracking and pose estimation

We used *faster-rcnn* to detect objects. We used *HOG+SVM* to obtain appearance similarity and *Kalman filter* to model motion dynamics. Summing up the scores from appearance model and motion model, we associate detections to tracking objects.  Then [*Convulutional Pose Machines*](https://github.com/shihenw/convolutional-pose-machines-release) was used to estimate wrist postions.

## Pre-requisites
- opencv 2.4.x
- caffe-faster-rcnn
- libconfig
- faster-rcnn model/weights && convolutional pose machines model/weights 

## Run-time libs
- export PYTHONPATH=/home/gengshan/workDec/threadProc/lib:/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/python
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gengshan/workDec/caffe-fast-rcnn-faster-rcnn/build/lib
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gengshan/workOct/cudnn_v3/lib64
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/softwares/libconfig/lib/

## Compile-time libs
- export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/cuda-7.0/pkgconfig

## Usage
- `make`
- clear output data folder `rm -rf data/*`
- run `./main  config.c`
- run genVideo.ipynb to generate example videos 

## TODO
- add something like sampleBgImg(bgImg), to sample foreground image
- ask how to get good samples in practice
- visualize sampled images
- make tracking better
- make pose better
- action recognition
- appearance model may be substituted by Siamese network for better performance  

## Acknowledgement
- faster-rcnn c++ implementation is based on [this blog](http://blog.csdn.net/xyy19920105/article/details/50440957)
- parameter configuration part is based on [this code](https://github.com/gnebehay/HoughTrack)

