#include "draw.hpp"

void objDraw::drawTracklet(Mat frame, unsigned int ID, 
                           vector<pair<unsigned int, unsigned int>> tracklet) const {
  for (unsigned int it = 1; it < tracklet.size(); it++) {
    /* Draw line */
    line(frame, Point(tracklet[it - 1].first, tracklet[it - 1].second),
                Point(tracklet[it].first, tracklet[it].second),
                colorArray[ID%colorNum], 5);
  }

  // imshow("tracklet", frame);
}
