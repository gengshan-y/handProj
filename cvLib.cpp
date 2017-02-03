#include "cvLib.hpp"

using namespace std;
using namespace cv;

/* Define a list variables */
/* for building detector */
const char* detectorPath = "./HogDetector.txt";  // const char* for input file 

/* global current frame to store results */
char countStr [50];
unsigned int currID = 0;

string appearancePath = "/data/gengshan/hdTracking/";
string outputPath = "/data/gengshan/handProj/";

void pauseFrame(unsigned int milliSeconds) {
    char key = (char) waitKey(milliSeconds);
    switch (key) {
    case 'q':
    case 'Q':
    case 27:
        exit(0);  // stop program=
    default:
        return;  // go on
    }
}

void buildDetector(HOGDescriptor& hog, const char* detectorPath) {
    /* Loading detector */
    /* loading file*/
    ifstream detFile(detectorPath, ios::binary);
    if (!detFile.is_open()) {
        cout << "HogDetector open failed." << endl;
        exit(-1);
    }
    detFile.seekg(0, ios_base::beg);

    vector<float> x;  // for constructing SVM
    float tmpVal = 0.0f;
    while (!detFile.eof()) {
        detFile >> tmpVal;
        x.push_back(tmpVal);
    }
    detFile.close();
    // cout << x.size() << " paramters loaded." << endl;

    /* constructing detector*/
    hog.setSVMDetector(x);
}

/* remove inner boxes */
vector<Rect> rmInnerBoxes(vector<Rect> found) {
    /* empty result */
    if (found.size() == 0) {
        return found;
    }

    /* with non-empty result */
    vector<Rect> foundFiltered;
    auto it = found.begin();
    auto itt = found.begin();
    for (it = found.begin(); it != found.end(); it++) {
        for (itt = found.begin(); itt != found.end(); itt++) {
            if (it != itt && ((*it & *itt) == *it) ) {
                break;
            }
        }
        if (itt == found.end()) {
            foundFiltered.push_back(*it);
        }
    }
    return foundFiltered;
}

TrackingObj measureObj(Mat targImg, Rect detRes) {
    Mat tmp;  // detected head img
    targImg.copyTo(tmp);
    return TrackingObj(currID++, tmp, detRes);  // measured object
                                                  // current ID is a faked one
}

void updateTracker(vector<Rect> found, Mat& targImg,
                   vector<TrackingObj>& tracker) {
    Mat oriImg;
    targImg.copyTo(oriImg);  // save original image

    /* Upgrade old tracking objects */
    for (auto it = tracker.begin(); it != tracker.end(); it++) {
        (*it).incAge();
        (*it).predKalmanFilter();
        (*it).showInfo();
    }

    /* Build measured objects */
    vector<TrackingObj> meaObjs;
    for (auto it = found.begin(); it != found.end(); it++)
        meaObjs.push_back(measureObj(targImg, *it));  // measured object
    // testStateParsing(meaObjs[0]);  // test the parsing interface

    /* Update/Add tracking objects */
    cout << "@@data association" << endl;
    vector<TrackingObj> lastTracker = tracker;  // save for association
    for (auto it = meaObjs.begin(); it != meaObjs.end(); it++) {
        /* get measured state */
        cout << "measured..." << endl;
        (*it).showState();
        vector<float> meaArray = (*it).getStateVec();

        vector<float> scoreArr;  // to store the comparison scores 
        for (auto itt = lastTracker.begin(); itt != lastTracker.end(); itt++) {
            /* get tracker predicted state */
            cout << "predicted..." << endl; 
            (*itt).showState();
            vector<float> predArray = (*itt).getStateVec();

            float score;
            // compare states
            float stateScore = meaStateDis(meaArray, predArray);
            // float alpha = 0.001; 
            // stateScore = exp(-stateScore * alpha);  // normalize to 0-1, tune a
            cout << "iou metric:\t" << stateScore << endl;;

            // get SVM score for measurement
            float SVMScore = (*itt).testSVM( (*it).getAppearance() );
            cout << "SVM score: \t" << SVMScore << endl;
            // waitKey(0);
            score = (stateScore + SVMScore) / 2.;

            scoreArr.push_back(score);  // add score to an array
        }

        unsigned int targIdx = distance(scoreArr.begin(),
                              max_element(scoreArr.begin(), scoreArr.end()));
        // unsigned int targIdx = distance(scoreArr.begin(),
        //                       min_element(scoreArr.begin(), scoreArr.end()));
        // if the highest score is higher than a th
        // if (scoreArr.size() != 0 && scoreArr[targIdx] < 1000) {
        if (scoreArr.size() != 0 && scoreArr[targIdx] > 0.5) {
            cout << "**ID " << tracker[targIdx].getID() << " updated" << endl;
            /* update the according tracker */
            // update tracklet
            tracker[targIdx].updateTracklet( (*it).getPos() );
            drawObj.drawTracklet(targImg, tracker[targIdx].getID(), 
                                 tracker[targIdx].getTracklet());
            // update SVM
            tracker[targIdx].updateSVM( oriImg, (*it).getAppearance() );
            tracker[targIdx].updateKalmanFilter( (*it).getMeaState() );
            tracker[targIdx].state2Attr();

            // reset age
            tracker[targIdx].resetAge();
            continue;
        }

        /* Else push detection results to tracker */
        // initialize tracklet
        (*it).initTracklet();
        // initialize SVM for *it
        (*it).initSVM(targImg);
        tracker.push_back(*it);
        currID++;  // update ID
        cout << "**ID " << tracker.back().getID() << " added." << endl;
        // tracker.back().showInfo();
    }

    /* Remove outdated objects */
    for (int it = tracker.size() - 1; it >= 0; it--) {
        if ( (tracker[it]).getAge() > 20 ) {  // set age as 20
            cout << "$$ID " << tracker[it].getID() << " to be deleted." << endl;
            
            // delete SVM for it
            (*(tracker.begin() + it)).rmSVM();

            // save appearance for future reference
            // (*(tracker.begin() + it)).svAppearance();
            destroyWindow("object " + \
               to_string((tracker.begin() + it)->getID()));  // destory window
            destroyWindow("pose " + \
               to_string((tracker.begin() + it)->getID()));
            tracker.erase(tracker.begin() + it);
        }
    }
}

void drawBBox(vector<Rect> found, Mat& targImg) {
    for (auto it = found.begin(); it != found.end(); it++){
        Rect r = *it;
        rectangle(targImg, r.tl(), r.br(), Scalar(0, 255, 0), 3);
    }
}

void extBBox(vector<Rect>& found) {
    for (unsigned int it = 0; it < found.size(); it++) {
        Rect r = found[it];
        // the HOG detector returns slightly larger rectangles
        // so we slightly shrink the rectangles to get a nicer output
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.9);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.9);
        found[it] = r;
    }
}

Mat combImgs(Mat img1, Mat img2) {
    Size sz1 = img1.size();
    Size sz2 = img2.size();
    Mat img3(sz1.height + 40, sz1.width + sz2.width, CV_8UC3,
             cvScalar(255, 255, 255));
    img1.copyTo(img3(Rect(0, 0, sz1.width, sz1.height)));
    img2.copyTo(img3(Rect(sz1.width, 0, sz2.width, sz2.height)));
    return img3;
}


float meaStateDis(vector<float> meaArray, vector<float> predArray) {
  float score = 0;
  // float posDist = pow(meaArray[0] - predArray[0], 2) +\ pow(meaArray[1] - predArray[1], 2);
  Rect r1(meaArray[0] - meaArray[4]/2., meaArray[1] - meaArray[5]/2., \
          meaArray[4], meaArray[5]);
  Rect r2(predArray[0] - predArray[4]/2., predArray[1] - predArray[5]/2., \
          predArray[4], predArray[5]);
  // cout << "r1:" << r1.x << "," << r1.y << "," << r1.width << "," << r1.height << endl;
  // cout << "r2:" << r2.x << "," << r2.y << "," << r2.width << "," << r2.height << endl;
  float iou = mMax(0, mMin(r1.x+r1.width, r2.x+r2.width) - mMax(r1.x, r2.x)) \
            * mMax(0, mMin(r1.y+r1.height, r2.y+r2.height) - mMax(r1.y, r2.y));
  // cout << "intersection:" << iou << endl;;
  score += iou / (r1.width * r1.height + r2.width * r2.height - iou);
  return score;
}

void testStateParsing(TrackingObj testObj) {
  TrackingObj tmpObj = testObj;
  tmpObj.attr2State();  // Flatten the attributes
  tmpObj.state2Attr();  // Fold the attributes
  if ( testObj == tmpObj ) {  // Make sure they are identical
      cout << "pass state parsing test" << endl;
  }
}


