#include "cvLib.hpp"

using namespace std;
using namespace cv;

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


