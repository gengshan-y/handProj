#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include "draw.hpp"
#include "parameters.hpp"
#include <opencv2/opencv.hpp>

/* Define interface for global vars */
extern const CvSize winSize;    //window size 
extern const CvSize blockSize;  //block size, fixed 
extern const CvSize blockStride;  //block stride, a multiple of cellSize 
extern const CvSize winStride;    //window stride, a multiple of blockStride 
extern const CvSize cellSize;     //cell size, fixed 
extern const int nbins;  // number of direction bins, fixed 

/* Init drawing object */
extern const objDraw drawObj;

/* For global configurations */
extern const Parameters *hp;

/* For image output */
extern string appearancePath;  // to store tracking object appearance
extern string outputPath;  // to store large output images

/* For tracking */
extern unsigned int currID;  // current object ID, declare with extern and 
                             // define in .cpp to avoid multiple definition

/* Others */
extern unsigned int frameCount;

#endif  // GLOBAL_HPP
