#include <iostream>
#include "global.hpp"
#include "Tracker.hpp"
#include "cvLib.hpp"
#include "caffeDet.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    /* Basic info */
    if (argc != 3) {
        cout << "./main input-vid-path " 
             << "display-result[y/n]" << endl;
        exit(-1);
    }
    cout << "OpenCV version " << CV_VERSION << endl;

    /* Initialization */
    Mat frame;  // to store video frames
    vector<Rect> found;  // to store detection results
    unsigned int count = 170;  // initialize the fist frame to be decoded, 80
    vector<TrackingObj> tracker;  // a tracker to monitor all heads

    /* Build detector */
    string model_file = "/home/gengshan/workDec/threadProc/model/faster_rcnn_test.pt";
    string weights_file = "/home/gengshan/workDec/threadProc/model/VGG16_faster_rcnn_final.caffemodel";
    int GPUID=0;
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    Detector caffeDet = Detector(model_file, weights_file);

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
       
        /* process a frame and get detection result */
        resize(frame, frame, imgSize);  // set to same-scale as train
        caffeDet.DetectImg(frame, found);

        Mat detFrame;
        frame.copyTo(detFrame);
        /* draw bounding box */
        drawBBox(found, detFrame);

        Mat trkFrame;
        frame.copyTo(trkFrame);
        /* get cropped images in detRests */
        updateTracker(found, trkFrame, tracker);
 
        /* save cropped image */
        // svCroppedImg(found, frame);
    
        /* show detection result */
        if (string(argv[2]) == "y") {
            imshow("demo", trkFrame);
            pauseFrame(0);
        }
        else {
            imwrite(outputPath + string(countStr) + ".jpg", trkFrame);
        }
    }
    
    return 0;
}
