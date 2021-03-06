#ifndef CAFFE_DET_HPP
#define CAFFE_DET_HPP

#include "gpu_nms.hpp"
#include "global.hpp"
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

const int class_num=21;
class Detector {
 public:
  /* Constructor */
  Detector(const string& model_file, const string& trained_file) {
    net_ = boost::shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(trained_file);
  }

  /* Detect objects in image and fill bounding boxes, return true if correct */
  bool DetectImg(cv::Mat& img, vector<cv::Rect>& found);

  /* Bounding box regression */
  void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
  void getBBox(vector<cv::Rect>& found, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH);
  void boxes_sort(int num, const float* pred, float* sorted_pred);
 private:
  boost::shared_ptr<Net<float> > net_;
  Detector(){}
};

struct myInfo
{
  float score;
  const float* head;
};

bool compInfo(const myInfo& myInfo1, const myInfo& myInfo2);

#endif  // CAFFE_DET_HPP
