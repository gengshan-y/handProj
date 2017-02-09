#ifndef CAFFE_POSE_HPP
#define CAFFE_POSE_HPP

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))

using namespace caffe;
using std::string;

class PoseMachine {
 public:
  /* Constructor */
  PoseMachine(const string& model_file, const string& trained_file) {
    net_ = boost::shared_ptr<Net<float> >(new Net<float>(model_file, 
                                                         caffe::TEST));
    net_->CopyTrainedLayersFrom(trained_file);
  }

  /* Estimate pose using the given model, withing the bounding box */
  void EstimateImgPara(vector<cv::Mat>& imgVec, vector<cv::Rect> rectVec, 
                       vector< std::tuple<int, int, float> > &resVec);

 private:
  boost::shared_ptr<Net<float> > net_;
  float *data_buf;  // net input
  vector<int> posXs;
  vector<int> posYs;
  vector<float> probs;

  /* Constructor */
  PoseMachine(){}

  /* Parse pose estimation results w.r.t. joint number */
  void parseRes(const float* source, float* dest, int jointNum);

  /** Preprocess image as did in matlab code, 
   ** including subtract and transpose
   */
  void Preprocess(cv::Mat img, float* data_buf);

  /* Add data_res to dataAggre */
  void dataSum(float* dataAggre, const float* data_res);

  /* Pad image tmp with given margins */
  void dataPad(cv::Mat cv_img, cv::Mat& padImg, int top, int down, int left, int right);

  /** get correspounding position of joints in the original image, 
   ** given the margin and scale.
   */
  void getJointPos(int imgNum, float *resJoint, 
                   int left, int top, float scale, 
                   vector< std::tuple<int, int, float> > &resVec);
};

#endif  // CAFFE_POSE_HPP
