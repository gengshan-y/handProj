#ifndef CV_LIB_HPP
#define CV_LIB_HPP

#include "draw.hpp"
#include "Tracker.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>


#define mMax(a, b) (((a)>(b)) ? (a) :(b))
#define mMin(a, b) (((a)<(b)) ? (a) :(b))


/* Pause current frame */
void pauseFrame(unsigned int milliSeconds);

/* remove inner boxes */
vector<Rect> rmInnerBoxes(vector<Rect> found);

/* build tracking object based on a detection result */
TrackingObj measureObj(Mat targImg, Rect detRes);

/* get cropped images from frame */
void updateTracker(vector<Rect> found, Mat& targImg, 
                   vector<TrackingObj>& tracker);

/* draw bounding box */
void drawBBox(vector<Rect> found, Mat& targImg);

/* Extend detection bounding box */
void extBBox(vector<Rect>& found);

/* Test sate parsing */
void testStateParsing(TrackingObj testObj);

/* combine two identical-sized images */
Mat combImgs(Mat img1, Mat img2);

/* measure state distance */
float meaStateDis(vector<float> meaArray, vector<float> predArray);

#endif  // CV_LIB_HPP
