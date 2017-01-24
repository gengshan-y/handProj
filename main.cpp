#include <iostream>
#include "global.hpp"
#include "Tracker.hpp"
#include "cvLib.hpp"
#include "caffeDet.hpp"
#include "caffePose.hpp"
#include <pthread.h>

using namespace std;
using namespace cv;

Mat frame;  // to store video frames
vector<Mat> frameArray;  // frame for parallism
vector<Rect> found;  // to store detection results
vector<TrackingObj> tracker;  // a tracker to monitor all heads
pthread_t detThread;
pthread_t poseThread;
pthread_mutex_t frameMutex;
pthread_mutex_t detMutex;
pthread_mutex_t trkMutex;
pthread_mutex_t poseMutex;

void *detFunc(void *args) {
    /* Build detector */
    string model_file = "/home/gengshan/workDec/threadProc/model/faster_rcnn_test.pt";
    string weights_file = "/home/gengshan/workDec/threadProc/model/VGG16_faster_rcnn_final.caffemodel";
    int GPUID=0;
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    Detector caffeDet = Detector(model_file, weights_file);

    while(1) {
        pthread_mutex_lock(&frameMutex);
        caffeDet.DetectImg(frame, found);
        pthread_mutex_unlock(&detMutex);
    }
    return NULL;
}

void *poseFunc(void *args) {
    /* Build pose estimator */
    int GPUID=0;
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    string model_file = "/home/gengshan/workDec/threadProc/model/pose_deploy_centerMap.prototxt";
    string weights_file = "/home/gengshan/workDec/threadProc/model/pose_iter_985000_addLEEDS.caffemodel";
    PoseMachine posMach = PoseMachine(model_file, weights_file);
    
    while(1) {
        pthread_mutex_lock(&trkMutex);
            /* get poes */
            for (auto it = tracker.begin(); it != tracker.end(); it++) {
                Mat posFrame;
                it->getFrame().copyTo(posFrame);
                Rect rect = it->getBBox();
                // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;
                cout << "@@pose " << it->getID() << endl;
                posMach.EstimateImg(posFrame, rect);  // for front3 199, very good for front pose
  
                // imshow("pose " + to_string(it->getID()), posFrame);
                imwrite(outputPath + "pose_" + to_string(it->getID()) + "_" + \
                        string(countStr) + ".jpg", posFrame);   
            }
        pthread_mutex_unlock(&poseMutex);
    }
}

int main(int argc, char* argv[]) {
    /* Basic info */
    if (argc != 3) {
        cout << "./main input-vid-path " 
             << "display-result[y/n]" << endl;
        exit(-1);
    }
    cout << "OpenCV version " << CV_VERSION << endl;

    /* Initialization */
    unsigned int count = 170;  // initialize the fist frame to be decoded, 80

 
    /* Create threads */
    pthread_mutex_lock(&frameMutex);
    pthread_mutex_lock(&detMutex);
    pthread_mutex_lock(&trkMutex);
    pthread_mutex_lock(&poseMutex);  // no resoures at first
    pthread_create(&detThread, NULL, detFunc, NULL);
    pthread_create(&poseThread, NULL, poseFunc, NULL);
 
    /* Read in frames and process */
    VideoCapture targetVid(argv[1]);
    if(!targetVid.isOpened()) {
        cout << "open failed." << endl;
        exit(-1);
    }
    unsigned int totalFrame = targetVid.get(CV_CAP_PROP_FRAME_COUNT);

    /* Set current to a pre-defined frame */
    targetVid.set(CV_CAP_PROP_POS_FRAMES, count);

    for(;;) {
        count++;
        targetVid >> frame;
        if(frame.empty()) {
            break;
        }

        /* get frame progress */
        sprintf(countStr, "%04d", count);  // padding with zeros
        cout << "------------------------------------------" 
             << "------------------------------------------" << endl;
        cout << "frame\t" << countStr << "/" << totalFrame << endl;
       
        /* process a frame */
        resize(frame, frame, imgSize);  // set to same-scale as train
        frameArray.push_back(frame);
        if (frameArray.size() < 10 != 0) {continue;}

        pthread_mutex_unlock(&frameMutex);  // signal detection
        
        /* draw bounding box
        Mat detFrame;
        frame.copyTo(detFrame);
        drawBBox(found, detFrame);
        */

        /* tracking */
        pthread_mutex_lock(&detMutex);  // wait for detection
        Mat trkFrame;
        frame.copyTo(trkFrame);
        updateTracker(found, trkFrame, tracker);
        pthread_mutex_unlock(&trkMutex);  // signal pose


        pthread_mutex_lock(&poseMutex);  // wait for pose
        /* save cropped image */
        // svCroppedImg(found, frame);
    
        /* show detection result */
        if (string(argv[2]) == "y") {
            imshow("demo", trkFrame);
            pauseFrame(1);
        }
        else {
            imwrite(outputPath + string(countStr) + ".jpg", trkFrame);
        }
    }
    
    return 0;
}
