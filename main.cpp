#include "cvLib.hpp"
#include "caffeDet.hpp"
#include "caffePose.hpp"
#include "global.hpp"
#include "parameters.hpp"
#include "Tracker.hpp"
#include <iostream>
#include <pthread.h>

using namespace std;
using namespace cv;

vector<Mat> frameArray;  // frame for parallism
vector<vector<Rect>> foundArray;
vector<Tracker> trackerArray;
pthread_t trkThread;
pthread_t poseThread;

pthread_mutex_t detSema;   // as semaphore
pthread_mutex_t trkSema;

pthread_mutex_t frameLock;  // as mutex
pthread_mutex_t trkLock;


void *trkFunc(void *args) {
  // Initialization
  Mat trkFrame;  // should be a new object if using push
  Tracker trk;  // keep track of all target objects

  // Tracking
  while(1) {
    Tracker tmpTrk;  // to store curr tracking res for disp
    pthread_mutex_lock(&detSema);  // wait for detection
    pthread_mutex_lock(&trkLock);  // lock trk result
    trackerArray.clear();
    for (unsigned int it = 0; it < frameArray.size(); it++) {
      trk.update(foundArray[it], frameArray[it]);  // update by detection
      tmpTrk = trk;
      trackerArray.push_back(tmpTrk);

      /* show tracking results */
      frameArray[it].copyTo(trkFrame);
      if (hp->readStringParameter("Conf.disp") == "y") {
        resize(trkFrame, trkFrame, Size(), 0.5, 0.5);
        imshow("tracking", trkFrame);
        pauseFrame(hp->readIntParameter("Conf.pauseMs"));
      }
      else {
        char countStr[50];
        sprintf(countStr, "%04d", frameCount);  // padding with zeros 
        imwrite(outputPath + "trk_" +  string(countStr) + ".jpg", trkFrame);
      }
    }

    pthread_mutex_unlock(&trkSema);  // signal pose
    pthread_mutex_unlock(&frameLock);  // signal fetch and det
  }
    return NULL;
}

void *poseFunc(void *args) {
    // Initialization
    Tracker tracker;  // temp tracker
    vector<Mat> posFrameVec;
    vector<Rect> rectVec;
    vector<int> idVec;
    vector< tuple<int, int, float> > resVec;  // results w.r.t each frame
    vector<unsigned int> frameNumVec;  // for safety
    Mat poseFrame;
    int batchSize = 1;

    // Build pose estimator
    Caffe::SetDevice(hp->readIntParameter("Conf.GPUID"));
    Caffe::set_mode(Caffe::GPU);
    string model_file = "/home/gengshan/workDec/threadProc/model/pose_deploy_centerMap.prototxt";
    string weights_file = "/home/gengshan/workDec/threadProc/model/pose_iter_985000_addLEEDS.caffemodel";
    PoseMachine posMach = PoseMachine(model_file, weights_file);
    
    while(1) {
        pthread_mutex_lock(&trkSema);
        posFrameVec.clear();
        rectVec.clear();
        idVec.clear();
        resVec.clear();
        frameNumVec.clear();  // for safety
   
        ofstream outFile("data/out.txt", ofstream::app);  // file writer

        // get image, position and ID
        for (unsigned int i = 0; i < trackerArray.size(); i++) {
            tracker = trackerArray[i];
            tracker.getBBox(posFrameVec, rectVec, idVec, frameNumVec);
            /*
            for (auto it = tracker.begin(); it != tracker.end(); it++) {
                Mat posFrame;
                it->getFrame().copyTo(posFrame);
                Rect rect = it->getBBox();
                // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;
                cout << "@@pose of ID " << it->getID() << endl;
                posFrameVec.push_back(posFrame);
                rectVec.push_back(rect);
                idVec.push_back(it->getID()); 
            }
            */
        }
 
        // should not change net size, so choose to pad
        while (rectVec.size() % batchSize != 0) {
            posFrameVec.push_back(posFrameVec[0]);
            rectVec.push_back(rectVec[0]);
            idVec.push_back(-1);  // mark as non object
            cout << "padded pose input" << endl;
        }

        // estimate poses
        unsigned int currHead = 0;
        while (currHead + batchSize - 1 < rectVec.size()) {
            vector<cv::Mat> tmpFrame(posFrameVec.begin() + currHead, 
                                posFrameVec.begin() + currHead + batchSize) ;
            vector<cv::Rect> tmpRect(rectVec.begin() + currHead, 
                                rectVec.begin() + currHead + batchSize);
            posMach.EstimateImgPara(tmpFrame, tmpRect, resVec);
            for (unsigned int it = 0; it < tmpFrame.size(); it++) {
                posFrameVec[currHead + it] = tmpFrame[it];
            }
            currHead += batchSize;
        }

        // show pose results
        for (unsigned int i = 0; i < posFrameVec.size() && idVec[i] >= 0; i++) {
                posFrameVec[i].copyTo(poseFrame);
                string title = "pose_" + to_string(idVec[i]);
                // draw results
                cv::circle(poseFrame, Point(get<0>(resVec[i]),   // left hand
                                            get<1>(resVec[i])),
                           1, cv::Scalar(0, 255, 0), 5);
                cv::circle(poseFrame, Point(get<0>(resVec[posFrameVec.size()+i]),
                                            get<1>(resVec[posFrameVec.size()+i])),
                           1, cv::Scalar(255, 0, 0), 5);  // right hand
              
                if (hp->readStringParameter("Conf.disp") == "y") {
                    resize(poseFrame, poseFrame, Size(), 0.5, 0.5);
                    imshow(title, poseFrame);
                    pauseFrame(hp->readIntParameter("Conf.pauseMs"));
                }
                else {
                    char countStr[50];
                    sprintf(countStr, "%04d", frameNumVec[i]);  // padding with zeros
                    imwrite(outputPath + title + "_" + string(countStr) + 
                            ".jpg", poseFrame);
                    outFile << title + "_" + string(countStr) << ", "
                            << get<0>(resVec[i]) << ", " 
                            << get<1>(resVec[i]) << ", "
                            << get<2>(resVec[i]) << "\n";
                }
        }
       
        outFile.close();
        pthread_mutex_unlock(&trkLock);  // signal tracking
    }
    return NULL;
}


int main(int argc, char* argv[]) {
    /* Basic info */
    if (argc != 2) {
        cout << "./main config-file" << endl;
        exit(-1);
    }
    cout << "OpenCV version " << CV_VERSION << endl;
    string paramFileName = string(argv[1]);    
    hp = new Parameters(paramFileName);

    /* Create threads */
    pthread_mutex_lock(&detSema);
    pthread_mutex_lock(&trkSema);
    pthread_create(&trkThread, NULL, trkFunc, NULL);
    pthread_create(&poseThread, NULL, poseFunc, NULL);

    /* Main thread for fetch and detection */
    // Initialization
    char countStr [50];
    vector<Rect> found;  // to store detection results

    // Build detector
    string model_file = "/home/gengshan/workDec/threadProc/model/faster_rcnn_test.pt";
    string weights_file = "/home/gengshan/workDec/threadProc/model/VGG16_faster_rcnn_final.caffemodel";
    Caffe::SetDevice(hp->readIntParameter("Conf.GPUID"));
    Caffe::set_mode(Caffe::GPU);
    Detector caffeDet = Detector(model_file, weights_file);

    // Read in frames and process
    VideoCapture targetVid(hp->readStringParameter("Conf.vidPath"));
    if(!targetVid.isOpened()) {
        cout << "open failed." << endl;
        exit(-1);
    }
    unsigned int totalFrame = targetVid.get(CV_CAP_PROP_FRAME_COUNT);

    // Set current to a pre-defined frame
    targetVid.set(CV_CAP_PROP_POS_FRAMES, frameCount);

    for(;;) {
        pthread_mutex_lock(&frameLock);  // lock frame and detection results
        frameArray.clear();  // clear before using
        foundArray.clear();  // clear rectangles

        // for a batch of 10 frames
        while (frameArray.size() < 1) {             
            Mat detFrame;  // frame to store detection results
            Mat frame;  // to store video frames, should be re-constructed
            frameCount++;
            targetVid >> frame;
            if(frame.empty()) {return -1;}

            // if (frameCount % 3 != 0) {continue;}  // skip some frames

            // get frame count
            sprintf(countStr, "%04d", frameCount);  // padding with zeros
            cout << "------------------------------------------" 
                 << "------------------------------------------" << endl;
            cout << "frame\t" << countStr << "/" << totalFrame << endl;
       
            // pre-process a frame
            resize(frame, frame, Size(), 1, 1);  // set to same-scale as train

            // detect and store
            if (!caffeDet.DetectImg(frame, found)) {
              cout << "detection error" << endl;
              exit(-1);
            }

            // add to buffer
            frameArray.push_back(frame);
            foundArray.push_back(found);
    
            // show detection results
            frame.copyTo(detFrame);            
            drawBBox(found, detFrame);
            if (hp->readStringParameter("Conf.disp") == "y") {
              resize(detFrame, detFrame, Size(), 0.5, 0.5);
              imshow("detection", detFrame);
              pauseFrame(hp->readIntParameter("Conf.pauseMs"));
            }
            else {
              imwrite(outputPath + "det_" +  string(countStr) + ".jpg", detFrame);
            }
        }
    
        pthread_mutex_unlock(&detSema);  // signal tracking thread
        
    }
    
    return 0;
}
