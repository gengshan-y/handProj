#include <iostream>
#include "global.hpp"
#include "Tracker.hpp"
#include "cvLib.hpp"
#include "caffeDet.hpp"
#include "caffePose.hpp"
#include <pthread.h>

using namespace std;
using namespace cv;

vector<Mat> frameArray;  // frame for parallism
vector<vector<Rect>> foundArray;
vector<vector<TrackingObj>> trackerArray;  // a tracker to monitor all heads
pthread_t detThread;
pthread_t trkThread;
pthread_t poseThread;
pthread_mutex_t frameMutex;
pthread_mutex_t detMutex;
pthread_mutex_t trkMutex;

pthread_mutex_t frameLock;
pthread_mutex_t detLock;
pthread_mutex_t trkLock;

string param;

void *detFunc(void *args) {
    /* Initialization */
    vector<Rect> found;  // to store detection results

    // Build detector
    string model_file = "/home/gengshan/workDec/threadProc/model/faster_rcnn_test.pt";
    string weights_file = "/home/gengshan/workDec/threadProc/model/VGG16_faster_rcnn_final.caffemodel";
    int GPUID=0;
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    Detector caffeDet = Detector(model_file, weights_file);

    while(1) {
        pthread_mutex_lock(&frameMutex);
        pthread_mutex_lock(&detLock);
        // detect and store
        foundArray.clear();  // clear rectangles
        for (auto it = frameArray.begin(); it != frameArray.end(); it++) {
            caffeDet.DetectImg(*it, found);
            foundArray.push_back(found);
        }     
        pthread_mutex_unlock(&detMutex);  // signal tracking
    }
    return NULL;
}

void *trkFunc(void *args) {
    
    /* Initialization */
    Mat trkFrame;  // should be a new object if using push
    vector<TrackingObj> tracker;  // a tracker to monitor all heads

    /* tracking */
    while(1) {
        pthread_mutex_lock(&trkLock);
        pthread_mutex_lock(&detMutex);  // wait for detection
        trackerArray.clear();
        for (unsigned int it = 0; it < frameArray.size(); it++) {
            frameArray[it].copyTo(trkFrame);
            updateTracker(foundArray[it], trkFrame, tracker);
            trackerArray.push_back(tracker);

            /* show tracking results */
            if (param == "y") {
                imshow("tracking", trkFrame);
                pauseFrame(1);
            }
            else {
                imwrite(outputPath + string(countStr) + ".jpg", trkFrame);
            }
        }

        pthread_mutex_unlock(&trkMutex);  // signal pose
        pthread_mutex_unlock(&detLock);  // signal det
        pthread_mutex_unlock(&frameLock);  // signal frame

    }
    return NULL;
}

void *poseFunc(void *args) {
    /* Initialization */
    vector<TrackingObj> tracker;  // a tracker to monitor all heads
    vector<Mat> posFrameVec;
    vector<Rect> rectVec;
    vector<int> idVec;

    /* Build pose estimator */
    int GPUID=0;
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    string model_file = "/home/gengshan/workDec/threadProc/model/pose_deploy_centerMap.prototxt";
    string weights_file = "/home/gengshan/workDec/threadProc/model/pose_iter_985000_addLEEDS.caffemodel";
    PoseMachine posMach = PoseMachine(model_file, weights_file);
    
    while(1) {
        pthread_mutex_lock(&trkMutex);
        posFrameVec.clear();
        rectVec.clear();
        idVec.clear();
        /* get poes */
        for (unsigned int i = 0; i < trackerArray.size(); i++) {
            tracker = trackerArray[i];
            for (auto it = tracker.begin(); it != tracker.end(); it++) {
                Mat posFrame;
                it->getFrame().copyTo(posFrame);
                Rect rect = it->getBBox();
                // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;
                cout << "@@pose " << it->getID() << endl;
                posFrameVec.push_back(posFrame);
                rectVec.push_back(rect);
                idVec.push_back(it->getID());        
            }
        }   
    
        /* should not change net size, so choose to pad */
        while (rectVec.size() % 5 != 0) {
            posFrameVec.push_back(posFrameVec[0]);
            rectVec.push_back(rectVec[0]);
            idVec.push_back(-1);  // mark as non object
            cout << "padded pose input" << endl;
        }

        unsigned int currHead = 0;
        while (currHead + 4 < rectVec.size()) {
            vector<cv::Mat> tmpFrame(posFrameVec.begin() + currHead, posFrameVec.begin() + currHead + 5) ;
            vector<cv::Rect> tmpRect(rectVec.begin() + currHead, rectVec.begin() + currHead + 5) ;
            posMach.EstimateImgPara(tmpFrame, tmpRect);
            for (unsigned int it = 0; it < tmpFrame.size(); it++) {
                posFrameVec[currHead + it] = tmpFrame[it];
            }
            currHead += 5;
        }
   
        for (unsigned int i = 0; i < posFrameVec.size(); i++) {
                /* show pose results */
                if (param == "y" && idVec[i] >= 0) {
                    imshow("pose " + to_string(idVec[i]), posFrameVec[i]);
                    pauseFrame(100);
                }
                else {
                    // imwrite(outputPath + string(countStr) + ".jpg", posFrame);
                }

                // imshow("pose " + to_string(it->getID()), posFrame);
                // imwrite(outputPath + "pose_" + to_string(it->getID()) + "_" + string(countStr) + ".jpg", posFrame);   
        }
       
        pthread_mutex_unlock(&trkLock);  // signal tracking
    }
    return NULL;
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
    Mat frame;  // to store video frames
    vector<Rect> found;  // to store detection results
    param = string(argv[2]);    
 
    /* Create threads */
    pthread_mutex_lock(&frameMutex);
    pthread_mutex_lock(&detMutex);
    pthread_mutex_lock(&trkMutex);
    pthread_create(&detThread, NULL, detFunc, NULL);
    pthread_create(&trkThread, NULL, trkFunc, NULL);
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
        pthread_mutex_lock(&frameLock);  // protect frameArray
        frameArray.clear();  // clear before using
        while (frameArray.size() < 10) {
            count++;
            targetVid >> frame;
            if(frame.empty()) {return -1;}

            if (count % 3 != 0) {continue;}

            /* get frame progress */
            sprintf(countStr, "%04d", count);  // padding with zeros
            cout << "------------------------------------------" 
                 << "------------------------------------------" << endl;
            cout << "frame\t" << countStr << "/" << totalFrame << endl;
       
            /* process a frame */
            resize(frame, frame, imgSize);  // set to same-scale as train
            frameArray.push_back(frame);
            cout << frameArray.size() << endl;
        }
        pthread_mutex_unlock(&frameMutex);  // signal detection
        
        /* draw bounding box
        Mat detFrame;
        frame.copyTo(detFrame);
        drawBBox(found, detFrame);
        */

        /* save cropped image */
        // svCroppedImg(found, frame);
    
    }
    
    return 0;
}
