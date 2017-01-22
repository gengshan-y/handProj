#ifndef DRAW_HPP
#define DRAW_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/** The class for drawing lines and rectangles
 */
class objDraw {
 public:
  /* Constructor */
  objDraw() {
    RNG rng(12345);
    colorNum = 100;
    for (unsigned int it = 0; it < colorNum; it++) {
      Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
      colorArray.push_back(color);
      /*
      Mat img(100, 100, CV_8UC3, color);
      cout << it << endl;
      imshow("", img);
      waitKey(0);
      */
    }
  }

  /* Draw tracklet of tracker in current frame */
  void drawTracklet(Mat frame, unsigned int ID, 
                    vector<pair<unsigned int, unsigned int>> tracklet) const;
  // should use const, since created as global const object, to avoid modification

 private:
  unsigned int colorNum;
  vector<Scalar> colorArray;
};

#endif  // DRAW_HPP
